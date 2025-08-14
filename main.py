import streamlit as st

# 페이지 제목
st.title("MBTI 기반 직업 추천")

# MBTI 목록
mbti_list = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
]

# MBTI별 추천 직업 데이터
mbti_jobs = {
    "INTJ": ["전략기획가", "데이터 과학자", "경영 컨설턴트"],
    "INTP": ["연구원", "프로그래머", "과학 작가"],
    "ENTJ": ["CEO", "프로젝트 매니저", "변호사"],
    "ENTP": ["창업가", "마케팅 전문가", "방송인"],
    "INFJ": ["심리상담가", "작가", "사회운동가"],
    "INFP": ["작가", "디자이너", "교사"],
    "ENFJ": ["교사", "HR매니저", "비영리단체 리더"],
    "ENFP": ["기획자", "광고 크리에이티브", "배우"],
    "ISTJ": ["회계사", "변호사", "경찰관"],
    "ISFJ": ["간호사", "교사", "사회복지사"],
    "ESTJ": ["경영자", "군인", "행정 공무원"],
    "ESFJ": ["간호사", "영업 관리자", "이벤트 플래너"],
    "ISTP": ["엔지니어", "정비사", "탐험가"],
    "ISFP": ["화가", "사진작가", "요리사"],
    "ESTP": ["영업사원", "스포츠 코치", "파일럿"],
    "ESFP": ["배우", "이벤트 기획자", "방송인"]
}

# 사용자 MBTI 입력
selected_mbti = st.selectbox("당신의 MBTI를 선택하세요", mbti_list)

# 추천 직업 출력
if selected_mbti:
    st.subheader(f"{selected_mbti} 추천 직업")
    jobs = mbti_jobs.get(selected_mbti, [])
    for job in jobs:
        st.write(f"- {job}")

