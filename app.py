import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date

from fib_pattern_engine_v5 import FibPatternEngineV5, train_and_save_model_v5
from auto_setup import fetch_setup

# ============================
# CONFIG
# ============================
st.set_page_config(
    page_title='Fib Path Analyzer V5',
    layout='wide',
    page_icon='📈',
    initial_sidebar_state='expanded',
)

BASE_DIR = Path(__file__).parent
EXCEL_PATH = BASE_DIR / 'Dataset Output Otomatis 2025-2026.xlsx'
MODEL_PATH = BASE_DIR / 'fib_pattern_engine_v5.pkl'
FIRST_HIT_SUMMARY_PATH = BASE_DIR / 'fib_pattern_first_hit_summary_v5.csv'
REACH_SUMMARY_PATH = BASE_DIR / 'fib_pattern_reach_summary_v5.csv'

ACTIONABLE_TARGETS = ['1.61_UP', '1.61_DOWN', '2.5_UP', '2.5_DOWN', '3.6_UP', '3.6_DOWN']
FIRST_HIT_TARGETS = ACTIONABLE_TARGETS + ['TIE_SAME_BAR', 'NO_HIT_48H']
CONTINUATION_ORDER = [
    'UP_1.61_TO_2.5', 'UP_2.5_TO_3.6', 'UP_1.61_TO_3.6',
    'DOWN_1.61_TO_2.5', 'DOWN_2.5_TO_3.6', 'DOWN_1.61_TO_3.6',
]

# Mapping warna momentum SQZMOM ke emoji buat visual cue
MOMENTUM_EMOJI = {'lime': '🟢', 'green': '🟩', 'red': '🟥', 'maroon': '🟫'}
SQUEEZE_EMOJI = {'Squeeze ON (black)': '⬛', 'Squeeze OFF (gray)': '⬜'}


# ============================
# HEADER
# ============================
st.title('📈 Fib Path Analyzer V5')
st.caption(
    'Engine k-NN similarity + exact-match pattern store. '
    'Setup di-fetch otomatis dari Binance (SQZMOM LazyBear + Fib Zone). '
    'User cukup input Trend.'
)


# ============================
# CACHING
# ============================
@st.cache_data
def get_unique_options(excel_path: Path):
    options = {'trends': ['Long', 'Short']}
    if not excel_path.exists():
        return options
    try:
        df = pd.read_excel(excel_path)
        if 'Trend' in df.columns:
            vals = sorted({str(x).strip() for x in df['Trend'].dropna() if str(x).strip()})
            if vals:
                options['trends'] = vals
    except Exception:
        pass
    return options


@st.cache_resource
def load_engine(model_path: Path):
    return FibPatternEngineV5.load(model_path)


# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.header('⚙️ Konfigurasi Model V5')

    if MODEL_PATH.exists():
        st.success('✅ Model V5 ready')
        retrain_label = '🔄 Retrain Model V5'
    else:
        st.warning('⚠️ Model V5 belum dilatih')
        retrain_label = '🚀 Train Model V5'

    if st.button(retrain_label, use_container_width=True):
        if EXCEL_PATH.exists():
            with st.spinner('Training engine V5 dari dataset...'):
                train_and_save_model_v5(
                    excel_path=EXCEL_PATH,
                    model_path=MODEL_PATH,
                    first_hit_summary_csv=FIRST_HIT_SUMMARY_PATH,
                    reach_summary_csv=REACH_SUMMARY_PATH,
                )
            load_engine.clear()
            get_unique_options.clear()
            st.success('Model V5 berhasil dilatih')
            st.rerun()
        else:
            st.error(f'Dataset tidak ditemukan: {EXCEL_PATH.name}')

    st.markdown('---')
    st.caption(
        '**Schema V5**: SQZMOM 1/2 (Value numeric + Momentum + Squeeze) '
        'menggantikan deskriptif "Rise weak white" di V2/V3. '
        'Fitur lain: Bar 1/2, Trend, Raw/Final Position, Score, Last TR.'
    )

    st.markdown('---')
    st.caption(f'Dataset: `{EXCEL_PATH.name}`')
    st.caption(f'Model: `{MODEL_PATH.name}`')


# ============================
# GATE
# ============================
if not MODEL_PATH.exists():
    st.info('👈 Train Model V5 dari sidebar dulu sebelum prediksi.')
    st.stop()

try:
    engine = load_engine(MODEL_PATH)
except Exception as e:
    st.error(f'Gagal load model V5: {e}')
    st.stop()

ops = get_unique_options(EXCEL_PATH)


# ============================
# INPUT FORM
# ============================
st.subheader('🎯 Setup Analisa')

with st.form('input_form'):
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.5])
    with f1:
        ticker_val = st.text_input('Ticker', value='ETH-USD',
                                   help='Format Yahoo: ETH-USD, BTC-USD, SOL-USD, ...')
    with f2:
        date_val = st.date_input('Tanggal (UTC)', value=date.today())
    with f3:
        hour_val = st.slider('Jam (UTC)', 0, 23, 0)
    with f4:
        trend_val = st.radio('Trend', ops['trends'] or ['Long', 'Short'], horizontal=True)

    submitted = st.form_submit_button('🔮 Jalankan Prediksi', use_container_width=True, type='primary')


# ============================
# SUBMIT FLOW
# ============================
if submitted:
    # Auto-fetch SEMUA field dari Binance (Bar + SQZMOM + Score + Posisi + Last TR)
    # supaya satu sumber data konsisten dengan dataset training.
    with st.spinner('📡 Fetch setup lengkap dari Binance...'):
        setup_auto = fetch_setup(ticker_val, date_val, hour_val)
    if setup_auto.get('error'):
        st.error(f"⚠️ Auto-fetch gagal: {setup_auto['error']}")
        st.stop()

    bar1_val = setup_auto['Bar 1']
    bar2_val = setup_auto['Bar 2']
    sqz1_mom = setup_auto['SQZMOM 1 Momentum']
    sqz1_sqz = setup_auto['SQZMOM 1 Squeeze']
    sqz1_val = setup_auto['SQZMOM 1 Value']
    sqz2_mom = setup_auto['SQZMOM 2 Momentum']
    sqz2_sqz = setup_auto['SQZMOM 2 Squeeze']
    sqz2_val = setup_auto['SQZMOM 2 Value']
    close_val = setup_auto.get('_close')

    score_val = setup_auto['Score']
    last_tr_val = setup_auto['Last TR']
    raw_pos_val = setup_auto['Raw Position']
    fin_pos_val = setup_auto['Final Position']

    # ============================
    # 🤖 AUTO-FETCH PANEL
    # ============================
    st.divider()
    st.subheader(f'🤖 Setup di {date_val} jam {hour_val:02d}:00 UTC ({ticker_val})')

    if close_val is not None:
        st.caption(f'Close price: **{close_val:,.2f}**')

    bar_col, sqz1_col, sqz2_col, sig_col = st.columns([1.1, 1.3, 1.3, 1.5])

    with bar_col:
        st.markdown('**🕯️ Bar**')
        st.metric('Bar 1 (jam ini)', bar1_val)
        st.metric('Bar 2 (jam lalu)', bar2_val)

    with sqz1_col:
        st.markdown(f'**📊 SQZMOM 1 (jam ini)** {MOMENTUM_EMOJI.get(sqz1_mom, "")} {SQUEEZE_EMOJI.get(sqz1_sqz, "")}')
        st.metric('Value', f'{sqz1_val:+.4f}')
        st.caption(f'Momentum: `{sqz1_mom}`')
        st.caption(f'Squeeze: `{sqz1_sqz}`')

    with sqz2_col:
        st.markdown(f'**📊 SQZMOM 2 (jam lalu)** {MOMENTUM_EMOJI.get(sqz2_mom, "")} {SQUEEZE_EMOJI.get(sqz2_sqz, "")}')
        st.metric('Value', f'{sqz2_val:+.4f}')
        st.caption(f'Momentum: `{sqz2_mom}`')
        st.caption(f'Squeeze: `{sqz2_sqz}`')

    with sig_col:
        st.markdown('**⚡ Signals (auto-compute)**')
        s1, s2 = st.columns(2)
        s1.metric('Score', f'{score_val}')
        s2.metric('Last TR', f'{last_tr_val:.2f}')
        s1.metric('Raw Position', raw_pos_val)
        s2.metric('Final Position', fin_pos_val)

    with st.expander('📋 Detail Indikator Market', expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric('Last Close', f"{setup_auto.get('last_close') or 0:.2f}")
        d2.metric('RSI 14', f"{setup_auto.get('rsi_last') or 0:.2f}")
        d3.metric('ADX 14', f"{setup_auto.get('adx_last') or 0:.2f}")
        d4.metric('ATR 14', f"{setup_auto.get('atr_last') or 0:.4f}")
        d1.metric('EMA 21', f"{setup_auto.get('ema_fast_last') or 0:.2f}")
        d2.metric('EMA 50', f"{setup_auto.get('ema_slow_last') or 0:.2f}")
        d3.metric('MACD', f"{setup_auto.get('macd_last') or 0:.4f}")
        d4.metric('Filter', setup_auto.get('filter_reason', '-'))

    # ============================
    # 🔮 PREDICT
    # ============================
    setup_data = {
        'Trend': trend_val,
        'SQZMOM 1 Momentum': sqz1_mom,
        'SQZMOM 1 Squeeze': sqz1_sqz,
        'SQZMOM 1 Value': sqz1_val,
        'SQZMOM 2 Momentum': sqz2_mom,
        'SQZMOM 2 Squeeze': sqz2_sqz,
        'SQZMOM 2 Value': sqz2_val,
        'Bar 1': bar1_val,
        'Bar 2': bar2_val,
        'Raw Position': raw_pos_val,
        'Final Position': fin_pos_val,
        'Score': score_val,
        'Last TR': last_tr_val,
    }

    with st.spinner('🧠 Menganalisis pola historis...'):
        result = engine.predict(setup_data, top_k_matches=5)

    # ============================
    # 📊 RESULT
    # ============================
    st.divider()
    st.header('📊 Hasil Prediksi')

    # Headline metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('1️⃣ First-hit utama',
              result.first_hit_top_target or '-', f'{result.first_hit_top_prob:.2%}')
    m2.metric('2️⃣ Kemungkinan kedua',
              result.first_hit_second_target or '-', f'{result.first_hit_second_prob:.2%}')
    m3.metric('⚠️ Risk Tie Same Bar', f'{result.tie_prob:.2%}')
    m4.metric('⌚ Risk No Hit 48h', f'{result.no_hit_prob:.2%}')

    reach_sorted = sorted(result.reach_probs.items(), key=lambda x: x[1], reverse=True)
    r1, r2 = st.columns(2)
    with r1:
        top_reach, top_reach_prob = (reach_sorted[0] if reach_sorted else ('-', 0.0))
        st.metric('🎯 Reach paling mungkin', top_reach, f'{top_reach_prob:.2%}')
    with r2:
        second_reach, second_reach_prob = (reach_sorted[1] if len(reach_sorted) > 1 else ('-', 0.0))
        st.metric('🎯 Reach kedua', second_reach, f'{second_reach_prob:.2%}')

    # Charts
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('##### Distribusi First-hit')
        first_hit_df = pd.DataFrame({
            'Target': FIRST_HIT_TARGETS,
            'Probabilitas (%)': [result.first_hit_probs.get(k, 0.0) * 100 for k in FIRST_HIT_TARGETS],
        }).set_index('Target')
        st.bar_chart(first_hit_df)

    with g2:
        st.markdown('##### Reach Probability Semua Fib')
        reach_df = pd.DataFrame({
            'Target': ACTIONABLE_TARGETS,
            'Probabilitas (%)': [result.reach_probs.get(k, 0.0) * 100 for k in ACTIONABLE_TARGETS],
        }).set_index('Target')
        st.bar_chart(reach_df)

    g3, g4 = st.columns([2, 1])
    with g3:
        st.markdown('##### Continuation Probability')
        cont_df = pd.DataFrame({
            'Transition': CONTINUATION_ORDER,
            'Probabilitas (%)': [result.continuation_probs.get(k, 0.0) * 100 for k in CONTINUATION_ORDER],
        }).set_index('Transition')
        st.bar_chart(cont_df)
    with g4:
        st.markdown('##### Sumber Keputusan')
        st.metric('Exact Match', f"{result.source_summary.get('exact_match_count', 0):.0f} data")
        st.metric('Bobot Exact', f"{result.source_summary.get('exact_weight_used', 0.0):.2%}")
        st.metric('Bobot Similarity', f"{result.source_summary.get('similarity_weight_used', 0.0):.2%}")

    # Probability tables
    with st.expander('📌 Tabel Probabilitas Lengkap', expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown('**First-hit probability**')
            first_hit_table = pd.DataFrame([
                {'Target': k, 'Probabilitas': v}
                for k, v in sorted(result.first_hit_probs.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(first_hit_table, use_container_width=True, hide_index=True)
        with right:
            st.markdown('**Reach probability semua fib**')
            reach_table = pd.DataFrame([
                {'Target': k, 'Probabilitas': v}
                for k, v in sorted(result.reach_probs.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(reach_table, use_container_width=True, hide_index=True)

        st.markdown('**Continuation probability**')
        cont_table = pd.DataFrame([
            {'Transition': k, 'Probabilitas': v}
            for k, v in sorted(result.continuation_probs.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(cont_table, use_container_width=True, hide_index=True)

    # Historical matches
    st.subheader('📚 Top Kasus Historis Paling Mirip')
    if result.top_matches:
        matches_df = pd.DataFrame(result.top_matches)
        rename_map = {
            'date': 'Tanggal', 'clock': 'Jam',
            'first_hit_target': 'First Hit', 'first_hit_direction': 'Arah',
            'first_hit_level': 'Level', 'reached_targets': 'Fib Tercapai',
            'similarity': 'Kemiripan', 'trend': 'Trend',
            'score': 'Score', 'last_tr': 'Last TR',
            'raw_position': 'Raw Position', 'final_position': 'Final Position',
        }
        matches_df.rename(columns=rename_map, inplace=True)
        if 'Tanggal' in matches_df.columns:
            matches_df['Tanggal'] = matches_df['Tanggal'].astype(str).replace('NaT', '-')
        if 'Kemiripan' in matches_df.columns:
            matches_df['Kemiripan'] = matches_df['Kemiripan'].apply(lambda x: f'{x:.2%}')
        show_cols = [
            'Tanggal', 'Jam', 'First Hit', 'Arah', 'Level', 'Fib Tercapai',
            'Kemiripan', 'Trend', 'Score', 'Last TR', 'Raw Position', 'Final Position',
        ]
        show_cols = [c for c in show_cols if c in matches_df.columns]
        st.dataframe(matches_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info('Tidak ada data kasus historis yang cocok.')
