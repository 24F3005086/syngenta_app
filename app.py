"""
Syngenta Field Force Intelligence
Flask Backend
May 17, 2026
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all CSVs"""
    try:
        data_dir = './data/'
        
        reps = pd.read_csv(f'{data_dir}reps_territory.csv')
        retailers = pd.read_csv(f'{data_dir}retailers.csv')
        visits = pd.read_csv(f'{data_dir}retailer_visit_log.csv')
        inventory = pd.read_csv(f'{data_dir}retailer_inventory_weekly.csv')
        pos = pd.read_csv(f'{data_dir}retailer_pos.csv')
        growers = pd.read_csv(f'{data_dir}growers.csv')
        digital = pd.read_csv(f'{data_dir}digital_funnel_weekly.csv')
        whatsapp = pd.read_csv(f'{data_dir}whatsapp_campaign.csv')
        
        # Convert dates
        visits['visit_date'] = pd.to_datetime(visits['visit_date'])
        inventory['week_end_date'] = pd.to_datetime(inventory['week_end_date'])
        pos['transaction_date'] = pd.to_datetime(pos['transaction_date'])
        
        return {
            'reps': reps,
            'retailers': retailers,
            'visits': visits,
            'inventory': inventory,
            'pos': pos,
            'growers': growers,
            'digital': digital,
            'whatsapp': whatsapp
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load data once at startup
DATA = load_data()

# ============================================================================
# SCORING ENGINE
# ============================================================================

class PrioritizationEngine:
    """Scores retailers based on 4 signals"""
    
    def __init__(self, data, today_date):
        self.data = data
        self.today_date = pd.to_datetime(today_date)
    
    def get_inventory_score(self, retailer_id):
        """Signal 1: Inventory Status (0-40 points)"""
        retailer_inv = self.data['inventory'][
            self.data['inventory']['retailer_id'] == retailer_id
        ].sort_values('week_end_date').tail(1)
        
        if retailer_inv.empty:
            return 5, "📦 No inventory data"
        
        inv = retailer_inv.iloc[0]
        sku_qty = inv['sku_qty']
        sku_name = inv['sku_name']
        
        if sku_qty == 0:
            return 40, f"🔴 OOS: {sku_name}"
        elif sku_qty < 5:
            return 25, f"🟡 Low stock: {sku_qty} units"
        else:
            return 5, f"🟢 Adequate: {sku_qty} units"
    
    def get_visit_recency_score(self, retailer_id, territory_id):
        """Signal 2: Visit Recency (0-25 points)"""
        territory_visits = self.data['visits'][
            self.data['visits']['territory_id'] == territory_id
        ].sort_values('visit_date')
        
        if territory_visits.empty:
            return 25, "📅 Territory never visited"
        
        last_visit = territory_visits.iloc[-1]['visit_date']
        days_since = (self.today_date - last_visit).days
        
        if days_since > 30:
            return 25, f"📅 Not visited in {days_since} days"
        elif days_since > 14:
            return 15, f"📅 Last visit {days_since} days ago"
        else:
            return min(int(days_since / 2), 10), f"📅 Recent ({days_since} days)"
    
    def get_sales_velocity_score(self, retailer_id):
        """Signal 3: Sales Velocity (0-20 points)"""
        four_weeks_ago = self.today_date - timedelta(days=28)
        
        retailer_sales = self.data['pos'][
            (self.data['pos']['retailer_id'] == retailer_id) &
            (self.data['pos']['transaction_date'] >= four_weeks_ago)
        ]
        
        if retailer_sales.empty:
            return 0, "📊 No sales last 4 weeks"
        
        total_qty = retailer_sales['sku_qty'].sum()
        avg_per_week = total_qty / 4
        score = min(int(avg_per_week / 5), 20)
        return score, f"📊 Velocity: {avg_per_week:.0f} units/week"
    
    def get_promotion_score(self, territory_id):
        """Signal 4: Promotion (0-15 points)"""
        last_week = self.today_date - timedelta(days=7)
        
        recent_promo = self.data['visits'][
            (self.data['visits']['visit_date'] >= last_week) &
            (self.data['visits']['territory_id'] == territory_id)
        ]
        
        if recent_promo.empty:
            return 5, "🎯 No recent promo"
        
        promo_count = len(recent_promo)
        return min(promo_count * 3, 15), f"🎯 {promo_count} promos last week"
    
    def score_retailer(self, retailer_id, territory_id):
        """Calculate composite score (0-100)"""
        inv_score, inv_reason = self.get_inventory_score(retailer_id)
        visit_score, visit_reason = self.get_visit_recency_score(retailer_id, territory_id)
        sales_score, sales_reason = self.get_sales_velocity_score(retailer_id)
        promo_score, promo_reason = self.get_promotion_score(territory_id)
        
        total_score = inv_score + visit_score + sales_score + promo_score
        total_score = min(total_score, 100)
        
        return {
            'retailer_id': retailer_id,
            'score': total_score,
            'breakdown': {
                'inventory': {'points': int(inv_score), 'reason': inv_reason},
                'visit_recency': {'points': int(visit_score), 'reason': visit_reason},
                'sales_velocity': {'points': int(sales_score), 'reason': sales_reason},
                'promotion': {'points': int(promo_score), 'reason': promo_reason}
            }
        }
    
    def get_top_retailers(self, territory_id, limit=10):
        """Get top N retailers"""
        territory_retailers = self.data['retailers'][
            self.data['retailers']['territory_id'] == territory_id
        ]['retailer_id'].unique()
        
        scores = []
        for retailer_id in territory_retailers:
            score = self.score_retailer(retailer_id, territory_id)
            scores.append(score)
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:limit]


class ActionRecommender:
    """Recommends next best action"""
    
    def __init__(self, data):
        self.data = data
    
    def recommend_action(self, retailer_id, today_date):
        """What to do at this retailer"""
        today_date = pd.to_datetime(today_date)
        
        retailer_inv = self.data['inventory'][
            self.data['inventory']['retailer_id'] == retailer_id
        ].sort_values('week_end_date').tail(1)
        
        if retailer_inv.empty:
            return {
                'action': 'MAINTAIN',
                'product': None,
                'reason': 'No inventory data',
                'urgency': 'LOW',
                'color': 'success'
            }
        
        inv = retailer_inv.iloc[0]
        sku_qty = inv['sku_qty']
        sku_name = inv['sku_name']
        
        if sku_qty == 0:
            return {
                'action': 'RESTOCK',
                'product': sku_name,
                'reason': 'Out of stock - URGENT',
                'urgency': 'CRITICAL',
                'color': 'danger'
            }
        
        if 0 < sku_qty < 5:
            return {
                'action': 'UPSELL',
                'product': sku_name,
                'reason': f'Low stock ({sku_qty} units)',
                'urgency': 'HIGH',
                'color': 'warning'
            }
        
        return {
            'action': 'MAINTAIN',
            'product': None,
            'reason': 'Inventory stable',
            'urgency': 'LOW',
            'color': 'success'
        }


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    if DATA is None:
        return render_template('index.html', error="Data not loaded. Check ./data/ folder.")
    
    rep_list = sorted(DATA['reps']['rep_id'].unique().tolist())
    return render_template('index.html', rep_list=rep_list, error=None)


@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    """API endpoint for recommendations"""
    try:
        data = request.json
        rep_id = data.get('rep_id')
        date_str = data.get('date')
        num_retailers = data.get('num_retailers', 10)
        
        # Get rep info
        rep_row = DATA['reps'][DATA['reps']['rep_id'] == rep_id]
        if rep_row.empty:
            return jsonify({'error': f'Rep {rep_id} not found'}), 404
        
        rep_info = rep_row.iloc[0]
        territory_id = rep_info['territory_id']
        
        # Score retailers
        scorer = PrioritizationEngine(DATA, date_str)
        top_retailers = scorer.get_top_retailers(territory_id, limit=num_retailers)
        
        # Get actions
        recommender = ActionRecommender(DATA)
        actions = {}
        for retailer in top_retailers:
            action = recommender.recommend_action(retailer['retailer_id'], date_str)
            actions[retailer['retailer_id']] = action
        
        return jsonify({
            'success': True,
            'rep_id': rep_id,
            'rep_name': rep_info.get('territory_name', 'N/A'),
            'state': rep_info.get('state', 'N/A'),
            'district': rep_info.get('district', 'N/A'),
            'territory_id': territory_id,
            'date': date_str,
            'num_retailers': len(top_retailers),
            'top_retailers': top_retailers,
            'actions': actions,
            'generated_at': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/<rep_id>/<date>', methods=['GET'])
def export_json(rep_id, date):
    """Export as JSON"""
    try:
        rep_row = DATA['reps'][DATA['reps']['rep_id'] == rep_id]
        if rep_row.empty:
            return jsonify({'error': 'Rep not found'}), 404
        
        rep_info = rep_row.iloc[0]
        territory_id = rep_info['territory_id']
        
        scorer = PrioritizationEngine(DATA, date)
        top_retailers = scorer.get_top_retailers(territory_id, limit=10)
        
        recommender = ActionRecommender(DATA)
        actions = {}
        for retailer in top_retailers:
            action = recommender.recommend_action(retailer['retailer_id'], date)
            actions[retailer['retailer_id']] = action
        
        export_data = {
            'rep_id': rep_id,
            'territory_id': territory_id,
            'date': date,
            'num_retailers': len(top_retailers),
            'top_retailers': top_retailers,
            'actions': actions,
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(export_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'OK',
        'app': 'Syngenta Field Force Intelligence',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', error="Page not found"), 404


@app.errorhandler(500)
def server_error(error):
    return render_template('index.html', error="Server error"), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    if DATA is None:
        print("❌ ERROR: Could not load data. Check ./data/ folder")
    else:
        print("✓ Data loaded successfully")
        print(f"✓ {len(DATA['reps'])} reps, {len(DATA['retailers'])} retailers")
        print("\n🚀 Starting Flask app...")
        print("📱 Open: http://localhost:5000")
        app.run(debug=True, port=5000)