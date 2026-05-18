from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    data = {}

    data['retailers'] = pd.read_csv('./data/retailers.csv')
    data['inventory'] = pd.read_csv('./data/retailer_inventory_weekly.csv')
    data['visits'] = pd.read_csv('./data/retailer_visit_log.csv')
    data['pos'] = pd.read_csv('./data/retailer_pos.csv')
    data['reps'] = pd.read_csv('./data/reps_territory.csv')

    data['inventory']['week_end_date'] = pd.to_datetime(data['inventory']['week_end_date'])
    data['visits']['visit_date'] = pd.to_datetime(data['visits']['visit_date'])
    data['pos']['transaction_date'] = pd.to_datetime(data['pos']['transaction_date'])

    return data

DATA = load_data()

# =========================================================
# HEALTH SCORE
# =========================================================

def calculate_health_score(retailer_id):
    sales = DATA['pos'][DATA['pos']['retailer_id'] == retailer_id]
    inventory = DATA['inventory'][DATA['inventory']['retailer_id'] == retailer_id]
    
    retailer_info = DATA['retailers'][DATA['retailers']['retailer_id'] == retailer_id]
    tehsil = retailer_info.iloc[0]['tehsil'] if not retailer_info.empty else None
    
    if tehsil:
        visits = DATA['visits'][DATA['visits']['visit_tehsil'] == tehsil]
    else:
        visits = pd.DataFrame()

    sales_score = 15
    if not sales.empty:
        last_date = sales['transaction_date'].max()
        recent_sales = sales[sales['transaction_date'] >= last_date - timedelta(days=30)]
        prev_sales = sales[(sales['transaction_date'] < last_date - timedelta(days=30)) & (sales['transaction_date'] >= last_date - timedelta(days=60))]
        rs_qty = recent_sales['sku_qty'].sum()
        ps_qty = prev_sales['sku_qty'].sum()
        
        if ps_qty > 0:
            growth = (rs_qty - ps_qty) / ps_qty
            if growth >= 0.1: sales_score = 30
            elif growth >= -0.1: sales_score = 25
            elif growth >= -0.5: sales_score = 15
            else: sales_score = 5
        else:
            sales_score = 25 if rs_qty > 0 else 0
            
    inventory_score = 25
    if not inventory.empty:
        latest_week = inventory['week_end_date'].max()
        latest_inv = inventory[inventory['week_end_date'] == latest_week]
        oos_count = len(latest_inv[latest_inv['sku_qty'] == 0])
        low_count = len(latest_inv[(latest_inv['sku_qty'] > 0) & (latest_inv['sku_qty'] < 5)])
        
        inventory_score -= (oos_count * 5)
        inventory_score -= (low_count * 2)
        inventory_score = max(0, inventory_score)
        
    visit_score = 0
    if not visits.empty:
        recent_visits = visits[visits['visit_date'] >= visits['visit_date'].max() - timedelta(days=90)]
        vc = len(recent_visits)
        if vc >= 5: visit_score = 20
        elif vc >= 2: visit_score = 15
        elif vc == 1: visit_score = 10
        
    campaign_score = 0
    if not visits.empty:
        campaigns = visits[visits['visit_type'] == 'campaign_conducted']
        if len(campaigns) >= 3: campaign_score = 25
        elif len(campaigns) >= 1: campaign_score = 15
        
    total = sales_score + inventory_score + visit_score + campaign_score

    if total >= 80:
        status = 'EXCELLENT'
        color = 'success'
    elif total >= 55:
        status = 'GOOD'
        color = 'warning'
    else:
        status = 'NEEDS ATTENTION'
        color = 'danger'

    churn_risk = 5
    if sales_score < 15: churn_risk += 35
    if inventory_score < 15: churn_risk += 20
    if visit_score == 0: churn_risk += 30
    churn_risk = min(95, churn_risk)

    return {
        'score': total,
        'status': status,
        'color': color,
        'churn_risk': churn_risk,
        'breakdown': {
            'sales': sales_score,
            'inventory': inventory_score,
            'visits': visit_score,
            'campaigns': campaign_score
        }
    }

# =========================================================
# AI PRIORITY ENGINE
# =========================================================

def get_priority(retailer_id):
    inventory = DATA['inventory']
    pos = DATA['pos']

    retailer_inv = inventory[inventory['retailer_id'] == retailer_id]
    retailer_sales = pos[pos['retailer_id'] == retailer_id]

    latest_inv = retailer_inv.tail(1)

    score = 0
    reason = []

    if not latest_inv.empty:
        qty = latest_inv.iloc[0]['sku_qty']

        if qty == 0:
            score += 40
            reason.append('Out of stock')

        elif qty < 5:
            score += 20
            reason.append('Low inventory')

    sales = retailer_sales['sku_qty'].sum()

    if sales > 100:
        score += 30
        reason.append('High sales velocity')

    return score, reason

# =========================================================
# CHATBOT
# =========================================================

import re

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    query = data.get('message', '').lower()

    # Advanced AI Simulation
    match = re.search(r'(rtl_\d+)', query)
    retailer_id = match.group(1).upper() if match else None

    if retailer_id:
        health = calculate_health_score(retailer_id)
        retailer_info = DATA['retailers'][DATA['retailers']['retailer_id'] == retailer_id]
        
        if not retailer_info.empty:
            territory = retailer_info.iloc[0]['territory_id']
            district = retailer_info.iloc[0]['district']
            
            if 'health' in query or 'score' in query:
                response = f"**{retailer_id}** is currently in **{health['status']}** status with a health score of **{health['score']}/100**. This score is based on recent POS data, inventory levels, and field visits in {district}."
            elif 'detail' in query or 'info' in query or 'who' in query or 'where' in query:
                response = f"**{retailer_id}** is located in district **{district}** (Territory: {territory}). Current health score is **{health['score']}**."
            else:
                response = f"For **{retailer_id}** (located in {district}), the health score is **{health['score']}** ({health['status']}). What specific metrics would you like to know (sales, inventory, or visits)?"
        else:
            response = f"I found the ID {retailer_id} but couldn't locate it in the retailers database."
            
    elif 'recommend' in query or 'priority' in query or 'top' in query:
        retailers = DATA['retailers']['retailer_id'].unique()[:5]
        response = "Based on my analysis of low inventory and high sales velocity, here are the top priority retailers to focus on:<br><br>"
        for r in retailers:
            score, _ = get_priority(r)
            if score > 0:
                response += f"• **{r}** (Priority Score: {score})<br>"
    elif 'sales' in query:
        response = "I track POS transaction data across all territories. To get sales insights, please ask about a specific retailer ID (e.g., 'What are the sales for RTL_00001?')."
    elif 'inventory' in query or 'stock' in query:
        response = "I monitor weekly inventory records. If you provide a retailer ID, I can check if they are running low or out of stock on key SKUs."
    elif 'hello' in query or 'hi' in query:
        response = "Hello! I am the Syngenta AI Copilot. I can analyze retailer health, prioritize restocking, and provide detailed insights. You can ask me things like:<br>• *'What is the health of RTL_00001?'*<br>• *'Which retailers are high priority?'*"
    else:
        response = "I can analyze retail intelligence data for you. Try asking about a specific retailer (e.g., 'Tell me about RTL_00001') or ask for recommendations."

    return jsonify({
        'response': response
    })

# =========================================================
# RETAILER DETAIL API
# =========================================================

@app.route('/api/retailer/<retailer_id>')
def retailer_detail(retailer_id):

    retailer = DATA['retailers'][
        DATA['retailers']['retailer_id'] == retailer_id
    ]

    inventory = DATA['inventory'][
        DATA['inventory']['retailer_id'] == retailer_id
    ].copy().tail(10)
    inventory['week_end_date'] = inventory['week_end_date'].astype(str)

    sales = DATA['pos'][
        DATA['pos']['retailer_id'] == retailer_id
    ]

    tehsil = retailer.iloc[0]['tehsil'] if not retailer.empty else None
    if tehsil:
        visits = DATA['visits'][DATA['visits']['visit_tehsil'] == tehsil].copy()
        visits['visit_date'] = visits['visit_date'].astype(str)
    else:
        visits = pd.DataFrame()

    health = calculate_health_score(retailer_id)

    total_sales = int(sales['sku_qty'].sum()) if not sales.empty else 0

    return jsonify({
        'retailer_id': retailer_id,
        'health_score': health,
        'total_sales': total_sales,
        'visit_count': len(visits),
        'inventory_records': inventory.to_dict(orient='records')
    })

# =========================================================
# =========================================================
# ANOMALY DETECTION, STATS, MAP & DIRECTORY
# =========================================================

def get_retailer_location(retailer_id, district):
    district_coords = {
        'Patna': (25.5941, 85.1376),
        'Jalgaon': (21.0077, 75.5626),
        'Vijayapura': (16.8302, 75.7100),
        'Ahmedabad': (23.0225, 72.5714),
        'Ludhiana': (30.9010, 75.8573),
        'Hisar': (29.1492, 75.7217),
        'Varanasi': (25.3176, 82.9739)
    }
    base_lat, base_lng = district_coords.get(district, (22.5, 78.5))
    h = int(hashlib.md5(retailer_id.encode()).hexdigest(), 16)
    lat_offset = (h % 1000) / 10000.0 - 0.05
    lng_offset = ((h // 1000) % 1000) / 10000.0 - 0.05
    return (base_lat + lat_offset, base_lng + lng_offset)

@app.route('/api/map-data')
def get_map_data():
    retailers = DATA['retailers'].head(200) # Load top 200 for demo
    
    map_data = []
    for _, r in retailers.iterrows():
        lat, lng = get_retailer_location(r['retailer_id'], r['district'])
        health = calculate_health_score(r['retailer_id'])
        
        map_data.append({
            'retailer_id': r['retailer_id'],
            'lat': lat,
            'lng': lng,
            'status': health['status'],
            'score': health['score']
        })
        
    return jsonify({'map_data': map_data})

@app.route('/api/locations')
def get_locations():
    # Returns nested dictionary { "Country": { "State": ["District1", "District2"] } }
    df = DATA['retailers']
    locations = {"India": {}}
    for state in df['state'].dropna().unique():
        districts = df[df['state'] == state]['district'].dropna().unique().tolist()
        locations["India"][state] = districts
    return jsonify(locations)

@app.route('/api/filter-retailers')
def filter_retailers():
    district = request.args.get('district')
    if not district:
        return jsonify([])
        
    filtered = DATA['retailers'][DATA['retailers']['district'] == district].head(15)
    results = []
    for _, r in filtered.iterrows():
        h = calculate_health_score(r['retailer_id'])
        results.append({
            'retailer_id': r['retailer_id'],
            'district': r['district'],
            'score': h['score'],
            'status': h['status'],
            'color': h['color']
        })
    return jsonify(results)

@app.route('/api/nearby-retailers')
def nearby_retailers():
    lat = float(request.args.get('lat', 22.5))
    lng = float(request.args.get('lng', 78.5))
    
    retailers = DATA['retailers'].head(300) # search pool
    results = []
    
    for _, r in retailers.iterrows():
        r_lat, r_lng = get_retailer_location(r['retailer_id'], r['district'])
        # Simple Euclidean distance for demo ranking
        dist = ((lat - r_lat)**2 + (lng - r_lng)**2)**0.5
        results.append({
            'retailer_id': r['retailer_id'],
            'district': r['district'],
            'distance': dist,
            'lat': r_lat,
            'lng': r_lng
        })
        
    results = sorted(results, key=lambda x: x['distance'])[:12]
    
    # Append health scores for top results
    for res in results:
        h = calculate_health_score(res['retailer_id'])
        res['score'] = h['score']
        res['status'] = h['status']
        res['color'] = h['color']
        
    return jsonify(results)

@app.route('/api/dashboard-stats')
def dashboard_stats():
    total_retailers = len(DATA['retailers'])
    
    pos = DATA['pos']
    pos_recent = pos[pos['transaction_date'] >= pos['transaction_date'].max() - timedelta(days=30)]
    revenue = (pos_recent['sku_qty'] * pos_recent['sku_price']).sum()
    
    inv = DATA['inventory']
    recent_date = inv['week_end_date'].max()
    oos_mask = (inv['week_end_date'] == recent_date) & (inv['sku_qty'] == 0)
    oos_retailers = inv[oos_mask]['retailer_id'].nunique()
    
    return jsonify({
        'total_retailers': total_retailers,
        'high_priority': int(total_retailers * 0.15),
        'critical_alerts': oos_retailers,
        'revenue_opportunity': float(revenue)
    })

@app.route('/api/alerts')
def get_alerts():
    alerts = []
    retailers = DATA['retailers']['retailer_id'].unique()[:100]
    
    for r in retailers:
        sales = DATA['pos'][DATA['pos']['retailer_id'] == r]
        if not sales.empty:
            last_date = sales['transaction_date'].max()
            recent_sales = sales[sales['transaction_date'] >= last_date - timedelta(days=30)]['sku_qty'].sum()
            prev_sales = sales[(sales['transaction_date'] < last_date - timedelta(days=30)) & (sales['transaction_date'] >= last_date - timedelta(days=60))]['sku_qty'].sum()
            if prev_sales > 0 and recent_sales < prev_sales * 0.5:
                alerts.append({
                    'type': 'Sales Drop',
                    'retailer_id': r,
                    'message': f'Sales dropped by >50% (from {prev_sales} to {recent_sales} units)',
                    'severity': 'High'
                })
        
        inventory = DATA['inventory'][DATA['inventory']['retailer_id'] == r]
        if not inventory.empty:
            dates = sorted(inventory['week_end_date'].unique())
            if len(dates) >= 2:
                last_2 = inventory[inventory['week_end_date'].isin(dates[-2:])]
                oos = last_2[last_2['sku_qty'] == 0]
                vc = oos['product_sku'].value_counts()
                for sku, count in vc.items():
                    if count >= 2:
                        alerts.append({
                            'type': 'OOS Streak',
                            'retailer_id': r,
                            'message': f'{sku} out of stock for 2+ weeks.',
                            'severity': 'High'
                        })
                        
        retailer_info = DATA['retailers'][DATA['retailers']['retailer_id'] == r]
        tehsil = retailer_info.iloc[0]['tehsil'] if not retailer_info.empty else None
        if tehsil:
            visits = DATA['visits'][DATA['visits']['visit_tehsil'] == tehsil]
            dataset_date = DATA['visits']['visit_date'].max()
            if visits.empty or visits['visit_date'].max() < dataset_date - timedelta(days=90):
                alerts.append({
                    'type': 'Inactive',
                    'retailer_id': r,
                    'message': 'No field visits in 90+ days.',
                    'severity': 'Medium'
                })
                    
    # Only return top alerts to avoid overwhelming UI
    return jsonify({'alerts': alerts[:12]})

@app.route('/api/forecast')
def seasonal_forecast():
    pos = DATA['pos']
    recent_date = pos['transaction_date'].max()
    month = recent_date.month
    
    if 6 <= month <= 10:
        season = "Kharif Season (Monsoon)"
        products = [
            {"product": "Score 250 EC", "crop": "Paddy/Rice", "demand": "HIGH", "insight": "High demand expected due to blast disease risk."},
            {"product": "Kavach 75 WP", "crop": "Vegetables", "demand": "HIGH", "insight": "Preventative fungicide for early blight."},
            {"product": "Amistar 250 SC", "crop": "Cotton", "demand": "MEDIUM", "insight": "Useful during square formation stage."},
            {"product": "Actara 25 WG", "crop": "Cotton", "demand": "HIGH", "insight": "Critical for sucking pest control."},
            {"product": "Cruiser 350 FS", "crop": "Soybean", "demand": "LOW", "insight": "Seed treatment phase is mostly over."}
        ]
    else:
        season = "Rabi Season (Winter)"
        products = [
            {"product": "Axial 50 EC", "crop": "Wheat", "demand": "HIGH", "insight": "Crucial for post-emergent grass weed control."},
            {"product": "Tilt 250 EC", "crop": "Wheat", "demand": "HIGH", "insight": "High risk of yellow rust; promote early sprays."},
            {"product": "Topik 15 WP", "crop": "Wheat", "demand": "MEDIUM", "insight": "Alternative herbicide for resistant weeds."},
            {"product": "Vertimec 1.8 EC", "crop": "Vegetables", "demand": "HIGH", "insight": "Mite infestations peak during dry spells."},
            {"product": "Movondo", "crop": "Mustard", "demand": "MEDIUM", "insight": "Effective against aphid attacks."}
        ]
        
    return jsonify({
        'season': season,
        'month': recent_date.strftime('%B %Y'),
        'forecast': products
    })

# =========================================================
# MAIN DASHBOARD API
# =========================================================

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():

    retailers = DATA['retailers']['retailer_id'].unique()[:10]

    output = []

    for retailer in retailers:
        score, reason = get_priority(retailer)
        health = calculate_health_score(retailer)

        output.append({
            'retailer_id': retailer,
            'priority_score': score,
            'reason': reason,
            'health': health
        })

    output = sorted(output, key=lambda x: x['priority_score'], reverse=True)

    return jsonify({
        'top_retailers': output
    })

# =========================================================
# PAGES
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/retailer/<retailer_id>')
def retailer_page(retailer_id):
    return render_template('retailer_detail.html', retailer_id=retailer_id)

# =========================================================
# RUN
# =========================================================

if __name__ == '__main__':
    app.run(debug=True, port=5001)