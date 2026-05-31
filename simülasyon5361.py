# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:46:55 2026

@author: Bthn5361
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Lineer Regresyon ve EKK Simülasyonu", page_icon="📈", layout="wide")

st.title("📈 Etkileşimli Lineer Regresyon: En Küçük Kareler (EKK)")
st.markdown("""
Bu simülasyon, **Basit ve Çoklu Doğrusal Regresyon** modellerinde parametre tahmin sürecini interaktif olarak göstermektedir. 
Aşağıdaki panelden parametreleri ayarlayarak, teorik En Küçük Kareler (OLS) çözümüne en yakın tahmini yapmaya çalışın. 
""")

with st.expander("📚 Teorik Arka Plan ve Formüller (Genel Gösterim)"):
    st.markdown("""
    **Doğrusal Regresyon Denklemleri:**
    
    * Basit Doğrusal Regresyon (2D): 
        
        $$y = \\beta_1 x + \\beta_0$$
        
    * Çoklu Doğrusal Regresyon (3D): 
        
        $$y = \\beta_1 x_1 + \\beta_2 x_2 + \\beta_0$$
        
    **Hata Kareler Ortalaması (MSE):** Modelin tahminleri ile gerçek değerler arasındaki farkların karelerinin ortalamasıdır. Hedef bu değeri minimize etmektir.
    """)
    st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
    st.markdown("**Belirlilik Katsayısı ($R^2$):** Bağımsız değişkenlerin, bağımlı değişkendeki varyansın ne kadarını açıkladığını gösterir. 1'e ne kadar yakınsa model o kadar iyidir.")
    st.latex(r"R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")

if "boyut" not in st.session_state:
    st.session_state["boyut"] = 1  
if "noise" not in st.session_state:
    st.session_state["noise"] = 15.0

def boyut_degistir(yeni_boyut):
    st.session_state["boyut"] = yeni_boyut
    st.session_state.pop("veri_sabit", None)

def gurultu_degisti():
    st.session_state.pop("veri_sabit", None)

def cozumu_goster_tetikle():
    st.session_state["cozumu_goster"] = True

def yeni_veri_tetikle():
    st.session_state.pop("veri_sabit", None)


st.markdown("### ⚙️ Veri Uzayı Seçimi")
col_dim1, col_dim2, _ = st.columns([1, 1, 4])

with col_dim1:
    st.button("2D (Tek Değişkenli)", on_click=boyut_degistir, args=(1,), use_container_width=True)

with col_dim2:
    st.button("3D (Çok Değişkenli)", on_click=boyut_degistir, args=(2,), use_container_width=True)

st.markdown("---")

if "veri_sabit" not in st.session_state:
    np.random.seed(np.random.randint(0, 1000))
    n_samples = 150
    noise = st.session_state["noise"] 
    
    if st.session_state["boyut"] == 1:
        X = np.random.rand(n_samples, 1) * 10
        gizli_egim = np.random.uniform(1.5, 5.0)
        gizli_kesisim = np.random.uniform(-10.0, 20.0)
        y = (gizli_egim * X[:, 0]) + gizli_kesisim + (np.random.randn(n_samples) * noise)
    else:
        X1 = np.random.rand(n_samples, 1) * 10
        X2 = np.random.rand(n_samples, 1) * 10
        X = np.hstack((X1, X2))
        gizli_egim1 = np.random.uniform(1.5, 5.0)
        gizli_egim2 = np.random.uniform(1.5, 5.0)
        gizli_kesisim = np.random.uniform(-10.0, 20.0)
        y = (gizli_egim1 * X[:, 0]) + (gizli_egim2 * X[:, 1]) + gizli_kesisim + (np.random.randn(n_samples) * noise)
    
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.session_state["n_samples"] = n_samples
    st.session_state["cozumu_goster"] = False
    st.session_state["veri_sabit"] = True

X = st.session_state["X"]
y = st.session_state["y"]
n_samples = st.session_state["n_samples"]
boyut = st.session_state["boyut"]
y_mean = np.mean(y)

st.subheader("📉 Parametre Uzayı Analizi")
col_ayarlar, col_grafik = st.columns([1.2, 2.8], gap="large")

with col_ayarlar:
    st.markdown("<br>", unsafe_allow_html=True)
    
    if boyut == 1:
        st.info("💡 Not: 2D uzayda regresyon doğrusu (çizgi) optimizasyonu yapılmaktadır.")
        user_slope = st.slider("Eğim (m)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        user_intercept = st.slider("Kesişim (b)", min_value=-30.0, max_value=50.0, value=0.0, step=1.0)
        y_user_pred = (user_slope * X[:, 0]) + user_intercept

    elif boyut == 2:
        st.info("💡 Not: 3D uzayda regresyon düzlemi (hiperdüzlem) optimizasyonu yapılmaktadır.")
        user_slope1 = st.slider("Eğim 1 (m1)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        user_slope2 = st.slider("Eğim 2 (m2)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        user_intercept = st.slider("Kesişim (b)", min_value=-30.0, max_value=50.0, value=0.0, step=1.0)
        y_user_pred = (user_slope1 * X[:, 0]) + (user_slope2 * X[:, 1]) + user_intercept

user_mse = mean_squared_error(y, y_user_pred)
user_r2 = r2_score(y, y_user_pred)

model = LinearRegression()
model.fit(X, y)
best_intercept = model.intercept_
y_best_pred = model.predict(X)

best_mse = mean_squared_error(y, y_best_pred)
best_r2 = r2_score(y, y_best_pred)

if boyut == 1:
    best_slope = model.coef_[0]
else:
    best_slope1 = model.coef_[0]
    best_slope2 = model.coef_[1]

with col_grafik:
    fig = go.Figure()

    if boyut == 1:
        X_plot = np.linspace(0, 10, 100)
        y_user_plot = (user_slope * X_plot) + user_intercept
        
        fig.add_trace(go.Scatter(x=X[:, 0], y=y, mode='markers', name='Gözlem Noktaları', marker=dict(color='royalblue', size=8, opacity=0.7)))
        fig.add_trace(go.Scatter(x=X_plot, y=y_user_plot, mode='lines', name='Sizin Tahmininiz', line=dict(color='firebrick', width=3)))

        if st.session_state["cozumu_goster"]:
            y_best_plot = (best_slope * X_plot) + best_intercept
            fig.add_trace(go.Scatter(x=X_plot, y=y_best_plot, mode='lines', name='EKK (En İyi) Çözümü', line=dict(color='seagreen', width=4, dash='dash')))

        y_min, y_max = y.min() - 10, y.max() + 10
        
        fig.update_layout(xaxis_title="Bağımsız Değişken (X)", yaxis_title="Bağımlı Değişken (Y)",
                          template="plotly_white", 
                          margin=dict(l=0, r=0, t=10, b=80), 
                          yaxis_range=[y_min, y_max],
                          font=dict(size=14), height=450,
                          legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))

    else:
        x1_mesh = np.linspace(0, 10, 20)
        x2_mesh = np.linspace(0, 10, 20)
        x1_grid, x2_grid = np.meshgrid(x1_mesh, x2_mesh)
        y_user_grid = (user_slope1 * x1_grid) + (user_slope2 * x2_grid) + user_intercept

        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y, mode='markers', name='Gözlem Noktaları', 
                                   marker=dict(size=4, color='royalblue', opacity=0.8)))
        
        fig.add_trace(go.Surface(x=x1_grid, y=x2_grid, z=y_user_grid, name='Sizin Tahmininiz',
                                 colorscale='Reds', showscale=False, opacity=0.8, showlegend=True))

        if st.session_state["cozumu_goster"]:
            # 3D Yüzeylerin göstergede (legend) çıkması için showlegend=True eklendi
            y_best_grid = (best_slope1 * x1_grid) + (best_slope2 * x2_grid) + best_intercept
            fig.add_trace(go.Surface(x=x1_grid, y=x2_grid, z=y_best_grid, name='EKK (En İyi) Düzlemi',
                                     colorscale='Greens', showscale=False, opacity=0.7, showlegend=True))

        fig.update_layout(scene=dict(xaxis_title='Değişken X1', yaxis_title='Değişken X2', zaxis_title='Hedef (Y)'),
                          template="plotly_white", 
                          margin=dict(l=0, r=0, t=10, b=80),
                          font=dict(size=12), height=500,
                          legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

col_btn1, col_slider, col_btn2 = st.columns([1.2, 2, 1.2], gap="medium")

with col_btn1:
    st.markdown("<br>", unsafe_allow_html=True) 
    st.button("✅ Optimizasyon Çözümünü Göster", on_click=cozumu_goster_tetikle, use_container_width=True)

with col_slider:
    st.slider("Veri Gürültüsü (Dağılım Miktarı)", min_value=0.0, max_value=50.0, step=1.0, key="noise", on_change=gurultu_degisti)

with col_btn2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔄 Yeni Veri Seti Üret", on_click=yeni_veri_tetikle, use_container_width=True)

st.markdown("---")
st.subheader("📊 Model Karşılaştırması")

col_user_metrik, col_best_metrik = st.columns(2, gap="large")

with col_user_metrik:
    st.markdown("🔴 **Sizin Kurduğunuz Model:**")
    
    c1, c2 = st.columns(2)
    c1.metric(label="Sizin MSE", value=f"{user_mse:.2f}")
    c2.metric(label="Sizin $R^2$", value=f"{user_r2:.3f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if boyut == 1:
        st.latex(f"\\hat{{y}} = {user_slope:.2f}x + ({user_intercept:.2f})")
    else:
        st.latex(f"\\hat{{y}} = {user_slope1:.2f}x_1 + {user_slope2:.2f}x_2 + ({user_intercept:.2f})")

with col_best_metrik:
    if st.session_state["cozumu_goster"]:
        st.markdown("🟢 **Optimizasyon Sonucu (EKK):**")
        
        c3, c4 = st.columns(2)
        fark_mse = best_mse - user_mse
        fark_r2 = best_r2 - user_r2
        
        c3.metric(label="Minimum MSE", value=f"{best_mse:.2f}", delta=f"{fark_mse:.2f} (Fark)", delta_color="inverse")
        c4.metric(label="Maksimum $R^2$", value=f"{best_r2:.3f}", delta=f"{fark_r2:.3f} (Fark)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if boyut == 1:
            st.latex(f"\\hat{{y}} = {best_slope:.2f}x + ({best_intercept:.2f})")
            
            st.latex(rf"""
            \begin{{aligned}}
            MSE &= \frac{{1}}{{{n_samples}}} \sum_{{i=1}}^{{{n_samples}}} \Big(y_i - ({best_slope:.2f}x_i + ({best_intercept:.2f}))\Big)^2 \\
            &= {best_mse:.2f}
            \end{{aligned}}
            """)
            
            st.latex(rf"""
            \begin{{aligned}}
            R^2 &= 1 - \frac{{\sum \Big(y_i - ({best_slope:.2f}x_i + ({best_intercept:.2f}))\Big)^2}}{{\sum (y_i - {y_mean:.2f})^2}} \\
            &= {best_r2:.3f}
            \end{{aligned}}
            """)
            
        else:
            st.latex(f"\\hat{{y}} = {best_slope1:.2f}x_1 + {best_slope2:.2f}x_2 + ({best_intercept:.2f})")
            
            st.latex(rf"""
            \begin{{aligned}}
            MSE &= \frac{{1}}{{{n_samples}}} \sum_{{i=1}}^{{{n_samples}}} \Big(y_i - ({best_slope1:.2f}x_{{1i}} + {best_slope2:.2f}x_{{2i}} + ({best_intercept:.2f}))\Big)^2 \\
            &= {best_mse:.2f}
            \end{{aligned}}
            """)
            
            st.latex(rf"""
            \begin{{aligned}}
            R^2 &= 1 - \frac{{\sum \Big(y_i - ({best_slope1:.2f}x_{{1i}} + {best_slope2:.2f}x_{{2i}} + ({best_intercept:.2f}))\Big)^2}}{{\sum (y_i - {y_mean:.2f})^2}} \\
            &= {best_r2:.3f}
            \end{{aligned}}
            """)
    else:
        st.info("🎯 Optimizasyon sonucunu ve karşılaştırmayı görmek için üstteki **'✅ Optimizasyon Çözümünü Göster'** butonuna tıklayın.")
