import streamlit as st

# 페이지 설정 🎨
st.set_page_config(
    page_title="MBTI 직업 추천 💼✨",
    page_icon="🧠",
    layout="centered"
)

# 타이틀 ✨
st.markdown("<h1 style='text-align: center; color: #ff6f61;'>💡 MBTI 기반 직업 추천 🎯</h1>", unsafe_allow_html=True)
st.markdown("---")

# MBTI 목록
mbti_list = [
    "INTJ 🦉", "INTP 🧩", "ENTJ 🦁", "ENTP 🎭",
    "INFJ 🌌", "INFP 🌸", "ENFJ 🌟", "ENFP 🔥",
    "ISTJ 📊", "ISFJ 🛡️", "ESTJ 🏛️", "ESFJ 🤝",
    "ISTP 🔧", "ISFP 🎨", "ESTP 🏎️", "ESFP 🎤"
]

# MBTI별 직업 데이터
mbti_jobs = {
    "INTJ": ["📊 전략기획가", "🧠 데이터 과학자", "💼 경영 컨설턴트"],
    "INTP": ["🔬 연구원", "💻 프로그래머", "📚 과학 작가"],
    "ENTJ": ["👑 CEO", "📅 프로젝트 매니저", "⚖️ 변호사"],
    "ENTP": ["🚀 창업가", "📣 마케팅 전문가", "🎥 방송인"],
    "INFJ": ["🗣️ 심리상담가", "✍️ 작가", "🌍 사회운동가"],
    "INFP": ["📖 작가", "🎨 디자이너", "🏫 교사"],
    "ENFJ": ["👩‍🏫 교사", "👥 HR매니저", "💖 비영리단체 리더"],
    "ENFP": ["💡 기획자", "🎬 광고 크리에이티브", "🎭 배우"],
    "ISTJ": ["📑 회계사", "⚖️ 변호사", "🚔 경찰관"],
    "ISFJ": ["🏥 간호사", "🏫 교사", "🤲 사회복지사"],
    "ESTJ": ["🏢 경영자", "🪖 군인", "📋 행정 공무원"],
    "ESFJ": ["🏥 간호사", "💼 영업 관리자", "🎉 이벤트 플래너"],
    "ISTP": ["🔧 엔지니어", "🛠️ 정비사", "🧭 탐험가"],
    "ISFP": ["🎨 화가", "📸 사진작가", "🍳 요리사"],
    "ESTP": ["📈 영업사원", "🏆 스포츠 코치", "✈️ 파일럿"],
    "ESFP": ["🎤 배우", "🎉 이벤트 기획자", "📺 방송인"]
}

# 선택 박스 🎯
selected = st.selectbox("👉 당신의 MBTI를 선택하세요!", mbti_list)

# MBTI 코드만 추출
mbti_code = selected.split(" ")[0]

# 결과 출력 🎉
if selected:
    st.markdown(f"## ✨ {selected}에게 어울리는 직업 추천 💼")
    st.markdown("---")
    jobs = mbti_jobs.get(mbti_code, [])
    for job in jobs:
        st.markdown(f"✅ {job}")

    # 분위기용 밑줄과 문구
    st.markdown("---")
    st.markdown(
        f"<h3 style='color: #ff6f61;'>🌟 {selected}의 특별한 장점 🌟</h3>",
        unsafe_allow_html=True
    )

    strengths = {
        "INTJ": "📌 체계적 계획 + 전략적 사고",
        "INTP": "📌 창의적 분석 + 논리적 탐구",
        "ENTJ": "📌 리더십 + 실행력 폭발",
        "ENTP": "📌 아이디어 뱅크 + 도전 정신",
        "INFJ": "📌 깊은 통찰 + 따뜻한 리더십",
        "INFP": "📌 순수한 가치관 + 창의성",
        "ENFJ": "📌 공감 능력 + 리더십",
        "ENFP": "📌 에너지 넘침 + 창의 폭발",
        "ISTJ": "📌 성실함 + 책임감 최고",
        "ISFJ": "📌 헌신적 + 세심함",
        "ESTJ": "📌 조직력 + 추진력",
        "ESFJ": "📌 친화력 + 서비스 정신",
        "ISTP": "📌 문제 해결 능력 + 실용적",
        "ISFP": "📌 예술 감각 + 자유로움",
        "ESTP": "📌 모험심 + 순발력",
        "ESFP": "📌 사교성 + 에너지"
    }

    st.success(strengths.get(mbti_code, "장점 데이터 없음 😅"))

    # 마무리 메시지 🎁
    st.markdown("---")
    st.markdown(
        "<h4 style='text-align: center; color: #6c63ff;'>🚀 당신의 잠재력을 믿으세요! 세상은 당신을 기다립니다 💖</h4>",
        unsafe_allow_html=True
    )
