# --- 라이브러리 임포트 ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import matplotlib.font_manager as fm
from pathlib import Path
import random
import platform

# --- 기본 설정 ---
st.set_page_config(page_title="AI 성장 플레이북 📖", layout="wide")

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
ASSETS = Path("assets")

# --- Gemini API 키 설정 ---
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# --- 폰트 설정 ---
plt.rc("font", family="AppleGothic")
plt.rcParams["axes.unicode_minus"] = False
sns.set(style="whitegrid", font="AppleGothic")

# --- 이모티콘 함수 ---
def emoji(category):
    emojis = {
        "delivery": ["🚚", "🍱", "🛵", "📦"],
        "sns": ["📱", "💬", "📸", "🔥", "🎯"],
        "up": ["📈", "💹", "✨"],
        "down": ["📉", "🫤", "💤"],
        "cafe": ["☕", "🍰", "🪑", "📷"],
        "food": ["🍜", "🍱", "🥘", "🍣"],
        "default": ["💡", "🧠", "🎯"]
    }
    return random.choice(emojis.get(category, emojis["default"]))

# ------------------------------
# 상단 헤더 구성
# ------------------------------
logo_path = ASSETS / "shc_ci_basic_00.png"

col1, col2 = st.columns([2, 5])
with col1:
    st.image(str(logo_path), width=250)
with col2:
    st.markdown(
        "<h4 style='text-align:right; color:#444;'>2025 Big Contest · AI DATA 활용분야</h4>",
        unsafe_allow_html=True
    )

st.markdown("---")

# --- title ---
st.title("🤖 AI 전략 가이드")
st.subheader("데이터 기반 맞춤형 AI 마케팅 전략 리포트")

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data():
    DATA1_PATH = Path("data/data1.csv")
    DATA2_PATH = Path("data/data2.csv")
    DATA3_PATH = Path("data/data3.csv")

    for path in [DATA1_PATH, DATA2_PATH, DATA3_PATH]:
        if not path.exists():
            st.error(f"❌ 파일이 없습니다: {path}")
            st.stop()

    try:
        data1 = pd.read_csv(DATA1_PATH, encoding="utf-8-sig")
        data2 = pd.read_csv(DATA2_PATH, encoding="utf-8-sig")
        data3 = pd.read_csv(DATA3_PATH, encoding="utf-8-sig")

        for df in [data1, data2, data3]:
            df.columns = df.columns.str.strip()

        for df in [data1, data2, data3]:
            if "ENCODED_MCT" in df.columns:
                df["ENCODED_MCT"] = df["ENCODED_MCT"].astype(str)
            if "TA_YM" in df.columns:
                df["TA_YM"] = df["TA_YM"].astype(str)

        merged_12 = pd.merge(data1, data2, on="ENCODED_MCT", how="inner")
        merged_all = pd.merge(merged_12, data3, on=["ENCODED_MCT", "TA_YM"], how="inner")

        return merged_all

    except Exception as e:
        st.error(f"⚠️ 데이터 병합 중 오류 발생: {e}")
        st.stop()

# --- 데이터 불러오기 ---
data = load_data()

data.columns = data.columns.str.strip()
data["ENCODED_MCT"] = data["ENCODED_MCT"].astype(str).str.strip().str.upper()
data["DISPLAY_NAME"] = (
    data["MCT_NM"].astype(str).str.strip() + " · " +
    data["ENCODED_MCT"].astype(str).str.strip()
)

# --- 가맹점 입력 ---
code_input = st.text_input("🏪 가맹점 코드를 입력하세요", placeholder="예: MCT01234")

if not code_input:
    st.info("가맹점 코드를 입력하면 분석이 시작됩니다.")
    st.stop()

code_input = code_input.strip().upper()
store_data = data[data["ENCODED_MCT"].astype(str).str.strip().str.upper() == code_input]

if store_data.empty:
    st.error("❌ 해당 가맹점코드로 등록된 매장이 없습니다.")
    st.stop()
else:
    selected_display = store_data["DISPLAY_NAME"].iloc[0]
    st.subheader(f"🏪 선택된 가맹점: {selected_display}")

# --- 주요 지표 계산 ---
delivery_rate = np.nanmean(pd.to_numeric(store_data["DLV_SAA_RAT"], errors="coerce"))
avg_delivery = np.nanmean(pd.to_numeric(data["DLV_SAA_RAT"], errors="coerce"))

young_rate = np.nanmean([
    pd.to_numeric(store_data.get("M12_MAL_1020_RAT", pd.Series([0])), errors="coerce").mean(),
    pd.to_numeric(store_data.get("M12_MAL_30_RAT", pd.Series([0])), errors="coerce").mean(),
    pd.to_numeric(store_data.get("M12_FME_1020_RAT", pd.Series([0])), errors="coerce").mean(),
    pd.to_numeric(store_data.get("M12_FME_30_RAT", pd.Series([0])), errors="coerce").mean()
])

avg_young = np.nanmean([
    pd.to_numeric(data.get("M12_MAL_1020_RAT", pd.Series([0])), errors="coerce").mean(),
    pd.to_numeric(data.get("M12_MAL_30_RAT", pd.Series([0])), errors="coerce").mean(),
    pd.to_numeric(data.get("M12_FME_1020_RAT", pd.Series([0])), errors="coerce").mean(),
    pd.to_numeric(data.get("M12_FME_30_RAT", pd.Series([0])), errors="coerce").mean()
])

revisit = np.nanmean(pd.to_numeric(store_data["MCT_UE_CLN_REU_RAT"], errors="coerce"))
avg_revisit = np.nanmean(pd.to_numeric(data["MCT_UE_CLN_REU_RAT"], errors="coerce"))
store_type = str(store_data.get("HPSN_MCT_BZN_CD_NM", "업종불명").iloc[0])

avg_temp = np.nanmean(pd.to_numeric(store_data.get("평균기온", pd.Series([24.6])), errors="coerce"))
event_count = np.nanmean(pd.to_numeric(store_data.get("지역이벤트발생횟수", pd.Series([1])), errors="coerce"))

# --- 요약 테이블 ---
summary_data = pd.DataFrame({
    "지표명": ["배달 매출(%)", "20~30대 고객 비중(%)", "재방문율(%)", "평균 기온(°C)", "이벤트 횟수(회)"],
    "값": [delivery_rate, young_rate, revisit, avg_temp, event_count]
})
st.markdown("### 📋 주요 지표 요약")
st.dataframe(summary_data.style.format({"값": "{:.2f}"}), use_container_width=True)

# --- 한글 폰트 설정 ---
if platform.system() == "Windows":
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == "Darwin":
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# --- 그래프 ---
cols = st.columns(3)
with cols[0]:
    fig, ax = plt.subplots(figsize=(2,1.5))
    sns.barplot(x=["매장","평균"], y=[delivery_rate, avg_delivery], palette=["#6EC1E4","#FFB6A0"], ax=ax)
    ax.set_title("🚚 배달 비중", fontsize=9)
    st.pyplot(fig)

with cols[1]:
    fig, ax = plt.subplots(figsize=(2,1.5))
    sns.barplot(x=["20~30대","기타"], y=[young_rate, 100-young_rate], palette=["#FF8E9E","#D3D3D3"], ax=ax)
    ax.set_title("🎯 연령대", fontsize=9)
    st.pyplot(fig)

with cols[2]:
    fig, ax = plt.subplots(figsize=(2,1.5))
    sns.barplot(x=["매장","평균"], y=[revisit, avg_revisit], palette=["#8AE68A","#C0C0C0"], ax=ax)
    ax.set_title("🔁 재방문율", fontsize=9)
    st.pyplot(fig)

# --- 조건별 프롬프트 ---
prompts = []

if "카페" in store_type or "커피" in store_type:
    prompts.append(f"""
[☕ 카페 업종 마케팅 전략]
{emoji('cafe')} 당신은 **카페 전문 AI 마케팅 참모**입니다.
- 업종: {store_type}
- 재방문율: {revisit:.1f}%
- 주 고객층: {young_rate:.1f}%

**전략 제안**
1️⃣ '스터디로그 챌린지' : 공부/노트 인증 시 무료 쿠폰 제공  
2️⃣ 'OOTD 콘테스트' : 포토존 릴스 중 최다 좋아요에 상품  
3️⃣ 날씨 기반 추천 음료 시스템 운영  
""")

if revisit < avg_revisit:
    prompts.append(f"""
[🔁 재방문율 개선 전략]
{emoji('down')} 현재 재방문율이 평균({avg_revisit:.1f}%)보다 낮습니다.  
**빠른 회복 전략 제안**
- 7일 이내 재방문 고객 대상 100% 당첨 쿠폰  
- 스탬프 적립 5회 시 추가 적립 제공  
- 재방문 고객 전용 톡 채널 운영  
""")

if any(k in store_type for k in ["한식","분식","양식","중식","일식","요리","식당"]):
    prompts.append(f"""
[🍱 요식업 매출 전략]
{emoji('food')} 업종: {store_type}  
**추천 전략**
1️⃣ 배달 메뉴 차별화 (대표메뉴 세트화)  
2️⃣ 점심 구독권 프로모션  
3️⃣ 리뷰 작성 시 즉시 할인 쿠폰 발급  
""")

if delivery_rate < avg_delivery:
    prompts.append(f"""
[🚚 배달 매출 향상 전략]
{emoji('delivery')} 현재 배달 매출 {delivery_rate:.1f}% (평균 {avg_delivery:.1f}%)  
- 더운 날씨엔 냉음료 세트, 추운 날엔 국물 메뉴 중심 쿠폰 자동 발행  
- 지역 이벤트 시 '방구석 응원 세트' 노출  
- 배달비 프로모션 자동화 시스템 적용  
""")

if young_rate > avg_young:
    prompts.append(f"""
[📱 2030 고객 중심 SNS 홍보 전략]
{emoji('sns')} 20~30대 고객 비중 {young_rate:.1f}% (평균 {avg_young:.1f}%)  
- 인스타/틱톡 기반 ‘날씨형 릴스 콘텐츠’ 제작  
- 해시태그 챌린지 운영 ('#오늘의한잔', '#나의출근브이로그')  
- SNS 반응률 10% ↑ 시 방문전환율 약 8% 상승 예상  
""")

# --- AI 실행 버튼 ---
if st.button("🧠 AI 전략 리포트 생성"):
    st.markdown("## 💬 AI 전략 리포트 결과")

    if len(prompts) == 0:
        st.warning("❗ 현재 조건에 해당하는 마케팅 전략이 없습니다.")
    else:
        for i, prompt in enumerate(prompts):
            title = prompt.strip().splitlines()[0]
            st.markdown("<br><hr style='border:1px solid #ccc'><br>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#0073e6'>{title}</h4>", unsafe_allow_html=True)

            with st.spinner(f"AI 분석 중… ({i+1}/{len(prompts)})"):
                response = model.generate_content(prompt)

            if response and response.text:
                st.markdown(response.text, unsafe_allow_html=True)
            else:
                st.info("⚠️ 해당 항목에 대한 분석 응답이 없습니다.")