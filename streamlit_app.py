import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import xgboost as xgb
import plotly.graph_objects as go

# ========== 页面美化 ==========
st.set_page_config(page_title="⚽ 智能足球预测系统", layout="wide")
st.title("⚽ 职业级足球预测模型 (Poisson + XGBoost)")
st.markdown("基于历史数据与实时特征，自动计算胜平负概率与推荐。")

# ========== 模拟历史数据生成 ==========
@st.cache_data
def load_data():
    np.random.seed(42)
    teams = ["皇马", "巴萨", "曼城", "利物浦", "拜仁", "巴黎", "国米", "阿森纳"]
    data = pd.DataFrame({
        'home_team': np.random.choice(teams, 1000),
        'away_team': np.random.choice(teams, 1000),
        'home_goals': np.random.poisson(1.6, 1000),
        'away_goals': np.random.poisson(1.2, 1000),
        'home_xG': np.random.normal(1.5, 0.4, 1000),
        'away_xG': np.random.normal(1.1, 0.4, 1000),
        'home_red': np.random.choice([0,1], 1000, p=[0.92,0.08]),
        'away_red': np.random.choice([0,1], 1000, p=[0.92,0.08]),
        'fatigue': np.random.uniform(0.8, 1.2, 1000),
        'temp': np.random.normal(20, 5, 1000)
    })
    # 确保主客队不同
    data = data[data['home_team'] != data['away_team']]
    return data

data = load_data()

# ========== 模型训练（带缓存） ==========
@st.cache_resource
def train_model(df):
    # 计算泊松基准
    league_avg_home = df['home_goals'].mean()
    league_avg_away = df['away_goals'].mean()
    
    home_attack = df.groupby('home_team')['home_goals'].mean()
    away_defense = df.groupby('away_team')['away_goals'].mean()
    
    def calc_poisson(row):
        lam_h = (home_attack.get(row['home_team'], 1.5) / league_avg_home) * \
                (league_avg_home / away_defense.get(row['away_team'], 1.2)) * league_avg_home
        lam_a = (home_attack.get(row['away_team'], 1.2) / league_avg_away) * \
                (league_avg_away / away_defense.get(row['home_team'], 1.5)) * league_avg_away
        return lam_h, lam_a
    
    df[['lambda_h', 'lambda_a']] = df.apply(calc_poisson, axis=1, result_type='expand')
    df['poisson_diff'] = df['lambda_h'] - df['lambda_a']
    df['goal_diff'] = df['home_goals'] - df['away_goals']
    
    # XGBoost特征
    X = df[['home_xG', 'away_xG', 'home_red', 'away_red', 'fatigue', 'temp', 'poisson_diff']]
    y = df['goal_diff']
    
    model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05)
    model.fit(X, y)
    
    # 返回所需对象
    return model, home_attack, away_defense, league_avg_home, league_avg_away

model, home_attack, away_defense, avg_h, avg_a = train_model(data)

# ========== 预测函数 ==========
def predict_match(home, away, xg_h, xg_a, red_h, red_a, fatigue, temp):
    # 计算泊松预期
    lam_h = (home_attack.get(home, 1.5) / avg_h) * (avg_h / away_defense.get(away, 1.2)) * avg_h
    lam_a = (home_attack.get(away, 1.2) / avg_a) * (avg_a / away_defense.get(home, 1.5)) * avg_a
    poisson_diff = lam_h - lam_a
    
    # 构造输入
    input_df = pd.DataFrame({
        'home_xG': [xg_h], 'away_xG': [xg_a],
        'home_red': [red_h], 'away_red': [red_a],
        'fatigue': [fatigue], 'temp': [temp],
        'poisson_diff': [poisson_diff]
    })
    
    pred_diff = model.predict(input_df)[0]
    
    # 概率映射
    prob_h = 1 / (1 + np.exp(-(pred_diff - 0.3)))
    prob_d = np.exp(-0.4 * (pred_diff ** 2)) / 2.2
    prob_a = 1 - prob_h - prob_d
    total = prob_h + prob_d + prob_a
    
    return {
        'diff': pred_diff,
        'home_win': prob_h/total,
        'draw': prob_d/total,
        'away_win': prob_a/total
    }

# ========== 侧边栏输入 ==========
st.sidebar.header("⚙️ 比赛参数设置")
teams = sorted(data['home_team'].unique())
home_team = st.sidebar.selectbox("🏠 主队", teams, index=0)
away_team = st.sidebar.selectbox("✈️ 客队", teams, index=1)

col1, col2 = st.sidebar.columns(2)
with col1:
    xg_h = st.number_input("主队 xG (预期进球)", 0.0, 5.0, 1.8, 0.1)
    red_h = st.selectbox("主队红牌", [0, 1])
with col2:
    xg_a = st.number_input("客队 xG (预期进球)", 0.0, 5.0, 1.2, 0.1)
    red_a = st.selectbox("客队红牌", [0, 1])

fatigue = st.sidebar.slider("赛程密集度 (0.8=疲劳, 1.2=轻松)", 0.8, 1.2, 1.0)
temp = st.sidebar.slider("比赛温度 (°C)", 5, 35, 20)

# ========== 主界面展示 ==========
if st.sidebar.button("🔮 开始预测", type="primary"):
    if home_team == away_team:
        st.error("主队和客队不能相同！")
    else:
        result = predict_match(home_team, away_team, xg_h, xg_a, red_h, red_a, fatigue, temp)
        
        # 显示结果
        st.subheader(f"📊 {home_team} vs {away_team} 预测结果")
        
        # 概率条
        fig = go.Figure(data=[go.Bar(
            x=['主胜', '平局', '客胜'],
            y=[result['home_win'], result['draw'], result['away_win']],
            marker_color=['#2E86C1', '#F4D03F', '#E74C3C'],
            text=[f"{p:.1%}" for p in [result['home_win'], result['draw'], result['away_win']]],
            textposition='outside'
        )])
        fig.update_layout(title="胜平负概率分布", yaxis_title="概率", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 推荐与净胜球
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("预测净胜球", f"{result['diff']:.2f}")
        with col_b:
            best = max(result.items(), key=lambda x: x[1] if x[0]!='diff' else 0)
            rec = "主胜" if best[0]=='home_win' else ("平局" if best[0]=='draw' else "客胜")
            st.metric("推荐赛果", rec, delta=f"概率 {best[1]:.1%}")
        with col_c:
            st.metric("模型置信度", f"{max(result['home_win'], result['draw'], result['away_win']):.1%}")
        
        # 详细数据
        with st.expander("📋 查看模型详细参数"):
            st.json({
                "主队预期进球(xG)": xg_h,
                "客队预期进球(xG)": xg_a,
                "泊松基准差值": f"{result['diff'] - (xg_h - xg_a):.2f} (修正量)",
                "红牌影响": f"{'-' if red_h else '+'}主队 / {'-' if red_a else '+'}客队",
                "状态因子": fatigue
            })

st.sidebar.markdown("---")
st.sidebar.info("💡 说明：\n- xG 可从Whoscored或Opta获取\n- 红牌按实际停赛填1\n- 疲劳度按3天内是否双赛调整")
