from flask import Flask, render_template, request, redirect, session, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')          # backend sans interface graphique
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import shutil
from pathlib import Path
import json

app = Flask(__name__)
app.secret_key = 'pfa'

# Configuration MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # Ton utilisateur MySQL (souvent 'root')
app.config['MYSQL_PASSWORD'] = 'RealMadrid1902'  # Ton mot de passe MySQL (vide si tu en as pas mis)
app.config['MYSQL_DB'] = 'portefeuille_db'

mysql = MySQL(app)

@app.route('/')
def home():
    if 'username' in session:
        return redirect('/dashboard')
    return render_template('home.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    # Redirect to dashboard if already logged in
    if 'username' in session:
        return redirect('/dashboard')
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        cur = mysql.connection.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            mysql.connection.commit()
            flash("Inscription réussie !", "success")
            return redirect('/login')
        except Exception as e:
            if "Duplicate entry" in str(e):
                flash("Ce nom d'utilisateur existe déjà.", "error")
            else:
                flash("Une erreur est survenue. Veuillez réessayer.", "error")
        finally:
            cur.close()

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Redirect to dashboard if already logged in
    if 'username' in session:
        return redirect('/dashboard')
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s ", (username,) )
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[2], password):
            session['username'] = username  # <<<<< ici on enregistre la session
            return redirect('/dashboard')
        
        flash("Nom d'utilisateur ou mot de passe incorrect.", "error")
        return redirect('/login') 
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect('/login')

    cur = mysql.connection.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (session['username'],))
    user = cur.fetchone()
    user_id = user[0]

    # Récupérer tous les portefeuilles de cet utilisateur
    cur.execute("""
        SELECT  p.id,
                p.name,
                COUNT(pos.id)                                           AS nb_positions,
                COALESCE(SUM(pos.quantity * pos.price_per_unit), 0)    AS valeur
        FROM portfolios           AS p
        LEFT JOIN positions  AS pos ON pos.portfolio_id = p.id
        WHERE p.user_id = %s
        GROUP BY p.id, p.name
        ORDER BY p.name;
    """, (user_id,))
    portfolios = cur.fetchall()
    portfolio_count = len(portfolios)
    cur.close()
    valeur_totale   = round(sum(p[3] for p in portfolios), 2)

    return render_template('dashboard.html', username=session['username'], portfolios=portfolios, portfolio_count=portfolio_count,valeur_totale=valeur_totale)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

@app.route('/create_portfolio', methods=['GET', 'POST'])
def create_portfolio():
    if 'username' not in session:
        return redirect('/login')
    
    if request.method == 'POST':
        portfolio_name = request.form['portfolio_name']

        # Trouver l'id de l'utilisateur connecté
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE username = %s", (session['username'],))
        user = cur.fetchone()
        user_id = user[0]

        # Insérer le nouveau portefeuille
        cur.execute("INSERT INTO portfolios (user_id, name) VALUES (%s, %s)", (user_id, portfolio_name))
        mysql.connection.commit()
        cur.close()

        return redirect('/dashboard')
    
    return render_template('create_portfolio.html')

@app.route('/view_portfolio/<int:portfolio_id>')
def view_portfolio(portfolio_id):
    if 'username' not in session:
        return redirect('/login')

    # Charger les positions du portefeuille
    cur = mysql.connection.cursor()
    cur.execute("SELECT action_name, quantity, price_per_unit, id, dpa FROM positions WHERE portfolio_id = %s", (portfolio_id,))
    positions_raw = cur.fetchall()

    # Charger le nom du portefeuille
    cur.execute("SELECT name FROM portfolios WHERE id = %s", (portfolio_id,))
    portfolio = cur.fetchone()
    cur.close()

    agg = defaultdict(lambda: [0, 0.0, None, 0.0])   # qty, valeur tot, id, dpa
    for tkr, qty, price, row_id, dpa in positions_raw:
        agg[tkr][0] += qty
        agg[tkr][1] += qty * price
        agg[tkr][2]  = row_id           # n’importe lequel suffit
        agg[tkr][3]  = dpa              # on garde le dpa (ou max/avg si besoin)

    # on reconstitue une liste au même format que l’original
    positions_raw = [
        (tkr,
        total_qty,
        total_val/total_qty,           # prix moyen pondéré
        any_id,
        dpa)
        for tkr, (total_qty, total_val, any_id, dpa) in agg.items()
    ]


    # Calculs de base
    total_value = sum(pos[1] * pos[2] for pos in positions_raw)
    total_shares = sum(pos[1] for pos in positions_raw)

    # Charger les rendements actions
    returns_df = pd.read_csv('static/stocks_returns.csv')
    years = ['2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024']

    returns_df['Rendement Arithmétique (%)'] = returns_df[years].mean(axis=1)
    returns_df['Rendement Géométrique (%)'] = ((1 + returns_df[years]).prod(axis=1)**(1/len(years)) - 1)

    # Charger les rendements du marché
    market_returns = pd.read_csv('static/market_returns.csv')
    market_returns['Année'] = market_returns['Année'].astype(str)   
    market_returns.set_index('Année', inplace=True)

    # Tickers présents
    tickers_all = [pos[0] for pos in positions_raw]
    returns_df = returns_df[returns_df['Ticker'].isin(tickers_all)]

    # Liste enrichie
    positions = []
    taux_croissance = 0.05  # pour Gordon-Shapiro
    for pos in positions_raw:
        ticker = pos[0]
        valeur_totale = pos[1] * pos[2]
        poids = valeur_totale / total_value if total_value > 0 else 0
        dpa = pos[4] or 0
        dividende_total = dpa * pos[1]
        rendement_dividende = (dpa / pos[2]) if pos[2] > 0 else None

        match = returns_df[returns_df['Ticker'] == ticker]
        if not match.empty:
            rentab_arith = match['Rendement Arithmétique (%)'].values[0]
            rentab_geo = match['Rendement Géométrique (%)'].values[0]
            ecart_type = match[years].std(axis=1).values[0]

            # Gordon-Shapiro
            V_gordon = dpa / (rentab_geo - taux_croissance) if dpa > 0 and rentab_geo > taux_croissance else None

            # Calcul bêta (avec alignement sur les années)
            # Calcul bêta (avec alignement correct)
            try:
                action_values = match[years].values.flatten().astype(float)
                market_values = market_returns.loc[years, 'Market'].values.astype(float)

                # Vérification des tailles et NaNs
                if len(action_values) == len(market_values):
                    df_beta = pd.DataFrame({'Action': action_values, 'Market': market_values}).dropna()
        
                    if not df_beta.empty and df_beta['Market'].var() > 0:
                        covariance = df_beta.cov().loc['Action', 'Market']
                        variance = df_beta['Market'].var()
                        beta = covariance / variance
                    else:
                        beta = None
                else:
                    beta = None
            except Exception as e:
                print(f"Erreur calcul bêta pour {ticker} : {e}")
                beta = None
        else:
            rentab_arith = rentab_geo = ecart_type = V_gordon = beta = None

        positions.append((
            pos[0], pos[1], pos[2], valeur_totale,
            rentab_arith, rentab_geo, pos[3],
            ecart_type, poids,
            dpa, dividende_total, rendement_dividende,
            V_gordon, beta
        ))
    
    labels_json = json.dumps([p[0] for p in positions])           # tickers
    values_json = json.dumps([round(p[3], 2) for p in positions]) # valeur € totale
    
    

    
    # Moyennes
    rentabilite_arithmetique_moyenne = returns_df['Rendement Arithmétique (%)'].mean()
    rentabilite_geometrique_moyenne = returns_df['Rendement Géométrique (%)'].mean()

    # Risque portefeuille
    rendements = returns_df.set_index('Ticker')[years].T
    covariance_matrix = rendements.cov()
    correlation_matrix = rendements.corr()

    # Supprimer les noms redondants d'axes
    correlation_matrix.columns.name = None
    correlation_matrix.index.name = None
    covariance_matrix.columns.name = None
    covariance_matrix.index.name = None
    
    ticker_to_value = {pos[0]: pos[1] * pos[2] for pos in positions_raw}
    filtered_values = [ticker_to_value[t] for t in covariance_matrix.columns if t in ticker_to_value]
    filtered_total = sum(filtered_values)
    poids_vecteur = np.array([v / filtered_total for v in filtered_values]) if filtered_total > 0 else np.zeros(len(filtered_values))
    variance_portefeuille = np.dot(poids_vecteur.T, np.dot(covariance_matrix.values, poids_vecteur))
    risque_portefeuille = np.sqrt(variance_portefeuille)

    # HTML matrices
    correlation_matrix_html = correlation_matrix.to_html(classes="table table-bordered", float_format="%.2f")
    covariance_matrix_html = covariance_matrix.to_html(classes="table table-bordered", float_format="%.4f")


    
    # Moyenne pondérée des rendements dividendes
    rendement_dividende_moyen = sum(pos[8] * pos[11] for pos in positions if pos[11] is not None)
    
    
    scatter_labels = [p[0] for p in positions]
    scatter_sigmas = [p[7] for p in positions]          # σ décimal
    scatter_mus    = [p[4] for p in positions]          # µ décimal

    pf_sigma = risque_portefeuille                      # σ portefeuille décimal
    pf_mu    = sum(p[8] * p[4] for p in positions if p[4] is not None)  # µ portefeuille décimal

    scatter_json = json.dumps({
        "labels": scatter_labels,
        "sigmas": [round(s*100, 2) if s is not None else None for s in scatter_sigmas],
        "mus":    [round(m*100, 2) if m is not None else None for m in scatter_mus],
        "pf": {"sigma": round(pf_sigma*100, 2), "mu": round(pf_mu*100, 2)}
    })
    
    # ─── Frontière efficiente ──────────────────────────────────────────────
    frontier_sigma = []
    frontier_mu    = []

    if len(covariance_matrix.columns) >= 2:          # il faut au moins 2 actions
        Σ   = covariance_matrix.values
        μ   = (returns_df
                .set_index('Ticker')
                .loc[covariance_matrix.columns, 'Rendement Arithmétique (%)']
                .values)
        ones = np.ones_like(μ)

        Σ_inv = np.linalg.inv(Σ)
        A = ones @ Σ_inv @ ones
        B = ones @ Σ_inv @ μ
        C = μ    @ Σ_inv @ μ
        D = A * C - B ** 2

        def min_var(mu_target):
            λ = (C - B * mu_target) / D
            γ = (A * mu_target - B) / D
            return Σ_inv @ (λ * ones + γ * μ)

        targets = np.linspace(min(μ), max(μ), 60)   # 60 points sur la frontière
        for m in targets:
            w = min_var(m)
            sigma = np.sqrt(w.T @ Σ @ w)
            frontier_sigma.append(round(float(sigma) * 100, 2))
            frontier_mu.append(round(float(m)     * 100, 2))

    else:
        # Moins de 2 titres -> pas de frontière
        frontier_sigma = frontier_mu = []

    # JSON pour Chart.js
    frontier_json = json.dumps({
        "sigma": frontier_sigma,
        "mu":    frontier_mu,
        "pf": {                               # point portefeuille (déjà calculé)
            "sigma": round(risque_portefeuille * 100, 2),
            "mu":    round(pf_mu * 100, 2)
        }
    })

    ############################################################################################
    has_positions = bool(positions_raw)
    if not positions_raw:                       # portefeuille vide
        flash("Votre portefeuille est encore vide !", "info")
        return render_template(
            "view_portfolio.html",
            positions=[],                        # table vide
            portfolio_name=portfolio[0],
            portfolio_id=portfolio_id,
            total_value=0,
            total_shares=0,
            rentabilite_arithmetique_moyenne=None,
            rentabilite_geometrique_moyenne=None,
            risque_portefeuille=None,
            correlation_matrix_html="",
            covariance_matrix_html="",
            rendement_dividende_moyen=None,
            graphs_available=False,               # indicateur pour le template
            labels_json="[]",
            values_json="[]",
            scatter_json="{}",
            frontier_json="{}",
            has_positions=False         
        )


   



    return render_template('view_portfolio.html',
                           positions=positions,
                           portfolio_name=portfolio[0],
                           portfolio_id=portfolio_id,
                           total_value=round(total_value, 2),
                           total_shares=total_shares,
                           rentabilite_arithmetique_moyenne=rentabilite_arithmetique_moyenne,
                           rentabilite_geometrique_moyenne=rentabilite_geometrique_moyenne,
                           risque_portefeuille=risque_portefeuille * 100,
                           correlation_matrix_html=correlation_matrix_html,
                           covariance_matrix_html=covariance_matrix_html,
                           rendement_dividende_moyen=rendement_dividende_moyen,
                           labels_json=labels_json,
                            values_json=values_json,
                            scatter_json=scatter_json,
                            frontier_json=frontier_json,
                            has_positions=True
                           )

 



@app.route('/add_position/<int:portfolio_id>', methods=['GET', 'POST'])
def add_position(portfolio_id):
    if 'username' not in session:
        return redirect('/login')

    if request.method == 'POST':
        action_name = request.form['action_name']
        quantity = int(request.form['quantity'])
        price_per_unit = float(request.form['price_per_unit'])
        dpa = request.form.get('dpa', type=float) or 0
        
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO positions (portfolio_id, action_name, quantity, price_per_unit,dpa) VALUES (%s, %s, %s, %s, %s)",
            (portfolio_id, action_name, quantity, price_per_unit,dpa)
        )
        mysql.connection.commit()
        cur.close()

        return redirect(f'/view_portfolio/{portfolio_id}')
    
    return render_template('add_position.html', portfolio_id=portfolio_id)

@app.route('/delete_portfolio/<int:portfolio_id>')
def delete_portfolio(portfolio_id):
    if 'username' not in session:
        return redirect('/login')

    cur = mysql.connection.cursor()

    # D'abord supprimer toutes les positions du portefeuille
    cur.execute("DELETE FROM positions WHERE portfolio_id = %s", (portfolio_id,))

    # Puis supprimer le portefeuille lui-même
    cur.execute("DELETE FROM portfolios WHERE id = %s", (portfolio_id,))

    mysql.connection.commit()
    cur.close()
    
    plots_dir = os.path.join(app.root_path, "static", "plots")
    for fname in ("pie_alloc.png", "scatter_rr.png", "efficient_frontier.png"):
        try:
            os.remove(os.path.join(plots_dir, fname))
        except FileNotFoundError:
            pass   
    flash('Portefeuille supprimé avec succès !')
    return redirect('/dashboard')

@app.route('/delete_position/<int:position_id>/<int:portfolio_id>')
def delete_position(position_id, portfolio_id):
    if 'username' not in session:
        return redirect('/login')
    
    cur = mysql.connection.cursor()
    cur.execute("SELECT action_name FROM positions WHERE id = %s", (position_id,))
    action_name=cur.fetchone()
    cur.execute("DELETE FROM positions WHERE action_name = %s and portfolio_id = %s ", (action_name,portfolio_id))
    mysql.connection.commit()
    cur.close()

    flash('Action supprimée avec succès !')  # ✅ Message flash
    return redirect(f'/view_portfolio/{portfolio_id}')

@app.route('/edit_position/<int:position_id>', methods=['GET', 'POST'])
def edit_position(position_id):
    if "username" not in session:
        return redirect("/login")

    cur = mysql.connection.cursor()

    # ─────────────────────────────
    # 1) récupérer la ligne à éditer
    cur.execute(
        "SELECT action_name, quantity, price_per_unit, dpa, portfolio_id "
        "FROM positions WHERE id = %s",
        (position_id,)
    )
    row = cur.fetchone()
    if row is None:
        flash("Action introuvable", "danger")
        return redirect("/dashboard")          # ou page d’erreur

    current_pos = {
        "ticker":         row[0],
        "quantity":       row[1],
        "price_per_unit": row[2],
        "dpa":            row[3] or 0
    }
    portfolio_id = row[4]                     # pour le lien « retour »
    # ─────────────────────────────

    # 2) si soumission du formulaire → mettre à jour
    if request.method == "POST":
        qty   = int(request.form["quantity"])
        price = float(request.form["price_per_unit"])
        dpa   = float(request.form["dpa"])

        cur.execute(
            "UPDATE positions SET quantity = %s, price_per_unit = %s, dpa = %s "
            "WHERE id = %s",
            (qty, price, dpa, position_id)
        )
        mysql.connection.commit()
        cur.close()

        flash("Action mise à jour !", "success")
        return redirect(f"/view_portfolio/{portfolio_id}")

    cur.close()
    # 3) affichage du formulaire pré-rempli
    return render_template(
        "edit_position.html",
        position=current_pos,
        portfolio_id=portfolio_id
    )



if __name__ == "__main__":
    app.run(debug=True)
