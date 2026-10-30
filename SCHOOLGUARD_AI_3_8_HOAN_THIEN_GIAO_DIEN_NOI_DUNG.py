import re
import random
import unicodedata
from io import StringIO

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="Phần mềm hỗ trợ học sinh THCS nhận diện nguy cơ và ứng xử an toàn trên không gian mạng",
    page_icon="🛡️",
    layout="wide"
)

def normalize(text):
    text = text.lower().replace('đ', 'd')
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

# Dữ liệu giả lập để hướng dẫn; không dùng làm chứng cứ độ chính xác thực tế.
RISK = [
    'Bạn đã trúng thưởng xe máy, hãy nộp phí nhận quà ngay',
    'Chuyển trước 200 nghìn phí xác nhận để nhận học bổng',
    'Nhận học bổng 5 triệu, đóng phí xử lý hồ sơ trước',
    'Tài khoản của bạn bị khóa, bấm liên kết để mở khóa ngay',
    'Cung cấp mã OTP để xác minh tài khoản',
    'Gửi mật khẩu đăng nhập cho nhân viên hỗ trợ',
    'Nhận quà miễn phí chỉ cần chuyển phí vận chuyển',
    'Bạn trúng điện thoại, thanh toán phí nhận thưởng',
    'Bấm link lạ để xác minh tài khoản ngân hàng',
    'Giả danh giáo viên yêu cầu chuyển tiền gấp',
    'Bạn được chọn nhận phần thưởng, nộp phí trước',
    'Nhập mã xác thực để nhận học bổng',
    'Chuyển khoản đặt cọc để nhận giải thưởng',
    'Tài khoản sắp bị vô hiệu hóa, đăng nhập tại liên kết này',
    'Gửi ảnh căn cước và mật khẩu để nhận quà',
    'Đóng 300 nghìn phí xác nhận học bổng hôm nay',
    'Nạp tiền vào ví để rút tiền thưởng',
    'Hãy gửi mã OTP ngay nếu không tài khoản sẽ bị khóa',
    'Nhận thưởng 10 triệu đồng nhưng phải trả phí trước',
    'Mở đường dẫn lạ và điền thông tin ngân hàng',
    'Giáo viên yêu cầu chuyển phí nhận học bổng vào tài khoản cá nhân',
    'Bạn trúng thưởng hãy bấm link và điền mật khẩu',
    'Nộp phí làm hồ sơ để nhận quà tặng',
    'Chuyển tiền ngay để không mất suất học bổng',
]
SAFE = [
    'Ngày mai lớp học môn Tin học tiết hai',
    'Cô giáo nhắc lớp nộp bài tập vào thứ hai',
    'Nhớ mang sách vở và bút đến lớp',
    'Nhóm mình họp làm bài thuyết trình vào chiều thứ sáu',
    'Lớp sẽ thực hành tại phòng máy vào sáng mai',
    'Bài học hướng dẫn không chia sẻ mật khẩu',
    'Giáo viên dặn không cung cấp mã OTP cho người khác',
    'Cả lớp tìm hiểu dấu hiệu tin nhắn trúng thưởng giả',
    'Hôm nay chúng ta học cách bảo vệ tài khoản',
    'Nhà trường tổ chức sinh hoạt dưới cờ',
    'Cô nhắc học sinh nộp phiếu học tập',
    'Các bạn nhớ xem lịch kiểm tra giữa kì',
    'Ngày mai mang tai nghe để học ngoại ngữ',
    'Lớp thảo luận về an toàn thông tin trên mạng',
    'Đừng bấm vào liên kết lạ và đừng chuyển tiền',
    'Hãy cảnh giác với lời mời nhận học bổng yêu cầu phí',
    'Bài tập: viết đoạn văn về phòng chống lừa đảo',
    'Mời phụ huynh dự họp tại trường vào chủ nhật',
    'Nhóm trực nhật đến sớm để vệ sinh lớp',
    'Bạn có thể gửi bài qua hệ thống học tập của trường',
    'Thầy nhắc không đăng mật khẩu lên mạng xã hội',
    'Ngày hội đọc sách tổ chức tại thư viện',
    'Chúc các bạn làm bài kiểm tra tốt',
    'Tiết học hôm nay nói về cách nhận biết tin giả',
]

@st.cache_resource
def train_model():
    model = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=1)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(RISK + SAFE, [1] * len(RISK) + [0] * len(SAFE))
    return model

model = train_model()




def rules(text):
    t = normalize(str(text))
    findings = []

    def has(*words):
        return any(w in t for w in words)

    protective = has(
        "khong yeu cau", "khong thu tien", "khong chia se",
        "khong cung cap", "khong gui", "khong nen",
        "khong duoc", "dung bam", "dung gui",
        "dung chuyen", "tuyet doi khong", "canh giac",
        "phong tranh", "phong chong", "bao nguoi lon"
    )
    learning = has(
        "bai hoc", "bai tap", "tiet hoc", "thao luan",
        "tim hieu", "tuyen truyen", "phan tich vi du",
        "nhan biet", "giai thich vi sao",
        "huong dan hoc sinh", "duoc huong dan",
        "gioi thieu nguy co", "nhac rang"
    )
    if (protective or learning) and not has(
        "nhung hay", "tuy nhien hay", "sau do hay"
    ):
        return ["Noi dung co dau hieu giao duc/phong tranh"], "education"

    # Rủ gặp riêng người quen qua mạng, giấu người thân: ưu tiên cảnh báo cao.
    online = has("quen qua facebook", "quen qua mang", "quen tren mang", "quen qua game", "nguoi quen qua facebook", "nguoi quen tren mang", "nguoi tren mang", "tren mang", "online")
    invite = has("ru em", "ru di", "ru hai dua", "ru chung em", "di uong nuoc", "di choi", "di xa", "di gap", "hen gap", "gap mat", "den nha", "toi nha")
    secrecy = has("khong ke voi bo me", "khong noi voi bo me", "khong cho bo me biet", "khong cho cha me biet", "dung cho bo me biet", "dung noi voi bo me", "giau bo me", "giau gia dinh", "giu bi mat", "khong ke voi gia dinh", "khong noi voi gia dinh")
    if online and invite and secrecy:
        findings.append("Du do gap rieng va giau gia dinh")

    if has("otp", "ma xac thuc", "ma dang nhap",
           "ma khoi phuc", "ma nhan hang") and has(
           "gui", "doc", "nhap", "cung cap", "chia se", "chup"):
        findings.append("Yeu cau ma xac thuc hoac ma khoi phuc")

    if has("mat khau", "so the", "thong tin ngan hang",
           "can cuoc", "giay khai sinh", "ma bao mat") and has(
           "gui", "nhap", "dien", "cung cap", "chup", "xac minh"):
        findings.append("Yeu cau thong tin nhay cam")

    if has("tat xac thuc", "tat bao mat",
           "tat phan mem diet virus"):
        findings.append("Yeu cau tat tinh nang bao mat")

    if has("chia se man hinh", "dieu khien tu xa",
           "ma qr dang nhap", "quet ma qr dang nhap"):
        findings.append("Nguy co chiem quyen truy cap")

    if has("apk", "tep thuc thi", "tep dinh kem") and has(
           "cai", "tai", "mo", "dinh kem"):
        findings.append("Phan mem hoac tep chua xac minh")

    if has("truy cap tin nhan sms", "cho phep truy cap tin nhan"):
        findings.append("Yeu cau quyen truy cap SMS")

    if has("duong dan rut gon", "lien ket la",
           "trang web khong thuoc", "trang xac thuc cua toi",
           "duong dan rieng") and has(
           "bam", "dang nhap", "nhap", "xac nhan", "truy cap"):
        findings.append("Lien ket hoac trang dang nhap dang nghi")

    if has("anh rieng tu", "ghi am rieng", "video rieng tu",
           "phat tan", "gui cho ca lop") and has(
           "chuyen tien", "tra tien", "neu khong", "doi tien"):
        findings.append("De doa, tong tien")

    if has("vi tri truc tiep", "lich o nha mot minh"):
        findings.append("Yeu cau thong tin vi tri rieng tu")

    if has("chuyen tien", "chuyen khoan", "nop phi",
           "dong phi", "dat coc", "thanh toan",
           "mua the cao", "tra tien") and has(
           "hoc bong", "qua", "thuong", "giai",
           "diem", "phi", "tai khoan ca nhan",
           "khan cap", "nguoi quan tri", "mo khoa"):
        findings.append("Yeu cau chuyen tien bat thuong")

    # Dấu hiệu khẩn cấp gây lo sợ kết hợp lời mời mở đường dẫn.
    # Không mặc định mọi liên kết đều nguy hiểm.
    urgent_event = has("tai nan", "nhap vien", "cap cuu", "nguy kich", "bi bat coc", "gap nan", "bi thuong")
    link_request = has("bam vao link", "bam link", "nhan vao link", "truy cap link", "bam vao duong link", "mo link", "bam vao lien ket")
    if urgent_event and link_request:
        findings.append("Tin khan cap kem lien ket can xac minh")

    if has("chan danh", "danh em", "danh chet", "de doa danh", "tung anh rieng tu", "phat tan anh", "ep gui anh rieng tu") and has("neu khong", "de doa", "se", "bat", "ep"):
        findings.append("De doa an toan ca nhan")
    return findings, "risk" if findings else "none"


def analyze(text):
    p = float(model.predict_proba([text])[0][1])
    findings, kind = rules(text)
    # Dùng cùng nhận diện nguy cơ rõ ràng với Góc chia sẻ để tránh bất nhất.
    if kind != 'education':
        advice_level, _, _ = advise_student_situation(text)
        if advice_level in ('NGUY CƠ CAO', 'CẦN HỖ TRỢ NGAY'):
            if not findings:
                findings.append('Nguy co tu tinh huong hoc sinh')
            kind = 'risk'
        elif advice_level in ('CẦN XÁC MINH', 'CẦN QUAN TÂM'):
            kind = 'caution' if kind != 'risk' else kind

    if kind == 'education':
        level = 'THẤP'
    elif kind == 'risk':
        level = 'CAO'
    elif kind == 'caution':
        level = 'TRUNG BÌNH'
    elif p >= 0.72:
        level = 'CAO'
    elif p >= 0.52:
        level = 'TRUNG BÌNH'
    else:
        level = 'THẤP'
    return p, level, findings

st.markdown("""<style>
.block-container {max-width: 1250px; padding-top: 1.6rem; padding-bottom: 2.5rem;}
html, body, [data-testid="stAppViewContainer"] {font-size: 20px !important;}
h1 {font-size: clamp(1.85rem, 3vw, 2.5rem) !important; line-height: 1.28 !important;}
h2 {font-size: clamp(1.75rem, 2.7vw, 2.15rem) !important;}
h3 {font-size: clamp(1.55rem, 2.3vw, 1.9rem) !important;}
.stMarkdown p, .stMarkdown li, [data-testid="stWidgetLabel"] p,
[data-testid="stAlert"] p, [data-testid="stCaptionContainer"] p
{font-size: 1.22rem !important; line-height: 1.55 !important;}
button, input, textarea, [role="radio"], [data-testid="stSelectbox"]
{font-size: 1.2rem !important;}
[data-testid="stTabs"] [role="tab"] {font-size: 1.24rem !important; font-weight: 700 !important; padding: 0.9rem 1.1rem !important; min-height: 3.6rem !important; white-space: nowrap !important;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {overflow-x: auto !important; gap: 0.25rem !important;}
[data-testid="stTextArea"] textarea {font-size: 1.23rem !important; line-height: 1.5 !important;}
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {font-size: 1.2rem !important; font-weight: 700; min-height: 3rem; border-radius: 9px;}
[data-testid="stAlert"] {border-radius: 10px;}
@media (max-width: 640px) {
 .block-container {padding-left: 0.8rem; padding-right: 0.8rem;}
 [data-testid="stTabs"] [role="tab"] {font-size: 1.12rem !important; padding: 0.7rem !important;}
}
</style>""", unsafe_allow_html=True)
st.title("PHẦN MỀM HỖ TRỢ HỌC SINH THCS NHẬN DIỆN NGUY CƠ VÀ ỨNG XỬ AN TOÀN TRÊN KHÔNG GIAN MẠNG")
st.caption('Nhận diện nguy cơ • Rèn kỹ năng • Bảo vệ bản thân')
st.caption('Phiên bản kiểm thử: 4.8 – ĐÃ BỔ SUNG NHẬN DIỆN DỤ DỖ QUA MẠNG')

tab1, tab2, tab3, tab6, tab5 = st.tabs(['🔎 Kiểm tra tin', '🎮 Thử thách', '🕵️ Thám tử an toàn mạng', '💬 Góc chia sẻ', 'ℹ️ Giới thiệu'])

with tab1:
    st.subheader('🔎 Kiểm tra mức độ an toàn của tin nhắn')
    message = st.text_area('Nhập tin nhắn cần kiểm tra', height=130, placeholder='Ví dụ: Em được nhận học bổng, hãy chuyển trước phí xác nhận...')
    if st.button('PHÂN TÍCH TIN NHẮN', type='primary'):
        if not message.strip():
            st.warning('Em hãy nhập một tin nhắn trước nhé!')
        else:
            p, level, findings = analyze(message)
            if level == 'CAO':
                st.error('🚨 Mức cảnh báo: CAO')
            elif level == 'TRUNG BÌNH':
                st.warning('⚠️ Mức cảnh báo: TRUNG BÌNH')
            else:
                st.success('🟢 Mức cảnh báo: THẤP')
            st.markdown('### 🔍 Vì sao tin nhắn được đánh giá như vậy?')
            if findings:
                descriptions = {
                    'Du do gap rieng va giau gia dinh': 'Người quen qua mạng rủ gặp ở nơi xa và yêu cầu giấu gia đình là dấu hiệu dụ dỗ nguy hiểm.',
                    'De doa an toan ca nhan': 'Tin nhắn có dấu hiệu đe dọa an toàn hoặc ép buộc.',
                    'Nguy co tu tinh huong hoc sinh': 'Nội dung có dấu hiệu nguy hiểm đối với học sinh; cần người lớn đáng tin cậy hỗ trợ.',
                    'Yeu cau ma xac thuc hoac ma khoi phuc': 'Tin nhắn yêu cầu mã OTP hoặc mã xác thực. Em tuyệt đối không đưa mã này cho người khác.',
                    'Yeu cau thong tin nhay cam': 'Tin nhắn yêu cầu thông tin riêng tư như mật khẩu, giấy tờ hoặc tài khoản ngân hàng.',
                    'Yeu cau tat tinh nang bao mat': 'Yêu cầu tắt tính năng bảo vệ thiết bị là dấu hiệu đáng cảnh giác.',
                    'Nguy co chiem quyen truy cap': 'Yêu cầu chia sẻ màn hình hoặc quyền điều khiển có thể khiến tài khoản bị chiếm quyền.',
                    'Phan mem hoac tep chua xac minh': 'Tệp hoặc ứng dụng chưa rõ nguồn gốc có thể gây nguy hiểm cho thiết bị.',
                    'Yeu cau quyen truy cap SMS': 'Quyền đọc SMS có thể làm lộ mã xác thực và thông tin riêng tư.',
                    'Lien ket hoac trang dang nhap dang nghi': 'Liên kết hoặc trang đăng nhập chưa xác minh có thể dùng để đánh cắp thông tin.',
                    'De doa, tong tien': 'Tin nhắn có dấu hiệu đe dọa hoặc ép buộc chuyển tiền.',
                    'Yeu cau thong tin vi tri rieng tu': 'Không nên chia sẻ vị trí hoặc lịch sinh hoạt riêng tư với người lạ.',
                    'Yeu cau chuyen tien bat thuong': 'Yêu cầu chuyển tiền để nhận quà, học bổng hoặc mở khóa tài khoản cần được xác minh.',
                    'Tin khan cap kem lien ket can xac minh': 'Tin nhắn kể chuyện tai nạn hoặc cấp cứu để gây lo lắng, rồi giục em mở đường link. Hãy gọi trực tiếp để xác minh trước.',
                    'Noi dung co dau hieu giao duc/phong tranh': 'Nội dung có dấu hiệu nhắc nhở hoặc giáo dục an toàn số; vẫn cần kiểm tra người gửi.'
                }
                for f in findings:
                    st.write('🔎 ' + descriptions.get(f, f))
            else:
                if level == 'THẤP':
                    st.write('Chưa phát hiện dấu hiệu đáng nghi rõ ràng như yêu cầu gửi mã OTP, mật khẩu hoặc chuyển tiền. Tuy vậy, em vẫn cần kiểm tra người gửi nhé!')
                else:
                    st.write('Tin nhắn chưa có dấu hiệu nguy hiểm rõ ràng, nhưng một số cách diễn đạt khiến AI cần cảnh báo. Em nên kiểm tra người gửi trước khi làm theo.')

            st.markdown('### 🛡️ Em nên làm gì tiếp theo?')
            if level == 'CAO':
                st.error('**DỪNG LẠI – KIỂM TRA TRƯỚC KHI HÀNH ĐỘNG**')
                st.write('① Không gặp riêng người quen qua mạng, không làm theo yêu cầu đe dọa hoặc giữ bí mật với gia đình.')
                st.write('② Không bấm liên kết hoặc tải tệp chưa xác minh.')
                st.write('③ Không gửi OTP, mật khẩu, thông tin cá nhân hay chuyển tiền.')
                st.write('④ Báo ngay cho cha mẹ, thầy cô hoặc người lớn đáng tin cậy; nếu có nguy hiểm trực tiếp, tìm trợ giúp khẩn cấp.')
            elif level == 'TRUNG BÌNH':
                st.warning('**CẦN XÁC MINH THÊM**')
                st.write('① Kiểm tra người gửi và nội dung qua kênh chính thức.')
                st.write('② Không cung cấp thông tin bảo mật hoặc chuyển tiền khi chưa xác minh.')
                st.write('③ Nếu còn phân vân, em hãy hỏi thầy cô hoặc cha mẹ.')
            else:
                st.success('**CHƯA THẤY DẤU HIỆU NGUY HIỂM RÕ RÀNG**')
                st.write('① Kiểm tra xem ai đã gửi tin nhắn.')
                st.write('② Chỉ làm theo yêu cầu sau khi biết rõ nguồn gửi.')
                st.write('③ Không chia sẻ mã OTP, mật khẩu hoặc thông tin riêng tư.')
                st.caption('Cảnh báo thấp không có nghĩa là tin nhắn chắc chắn an toàn.')
                st.caption('Đây là phần mềm hỗ trợ giáo dục, giúp học sinh nhận diện nguy cơ và rèn luyện kỹ năng ứng xử an toàn trên không gian mạng; không thay thế việc xác minh thực tế hoặc sự hỗ trợ của thầy cô, cha mẹ.')

QUIZ_BANK = [('Tin nhắn yêu cầu mã OTP để nhận học bổng, em nên làm gì?', ['Gửi mã OTP vì tin nhắn nhắc đến học bổng', 'Hỏi trực tiếp thầy cô để kiểm tra', 'Nhắn lại hỏi người gửi có đúng là cán bộ nhà trường không'], 1, 'Không chia sẻ mã OTP; hỏi thầy cô để xác minh.'), ('Em nhận được thông báo trúng thưởng nhưng phải nộp phí trước. Em nên làm gì?', ['Nộp phí nhỏ để giữ phần thưởng', 'Không chuyển tiền, kiểm tra lại thông tin', 'Hỏi người gửi xem có thể nhận thưởng sau không'], 1, 'Yêu cầu phí trước là dấu hiệu đáng nghi.'), ('Cô giáo nhắn nhắc lớp nộp bài tập. Em nên nhận xét thế nào?', ['Có thể là thông báo học tập', 'Chỉ tin khi có nhiều bạn chuyển tiếp', 'Cần đăng nhập qua đường liên kết trong tin nhắn'], 0, 'Thông báo học tập có thể bình thường, nhưng nên kiểm tra người gửi.'), ('Một đường liên kết lạ yêu cầu em đăng nhập tài khoản ngân hàng. Em nên làm gì?', ['Mở liên kết để xem giao diện trước', 'Không bấm đường liên kết, tự mở ứng dụng chính thức', 'Hỏi người gửi xem đường liên kết có an toàn không'], 1, 'Chỉ đăng nhập qua ứng dụng hoặc trang chính thức.'), ('Người lạ hỏi mật khẩu tài khoản của em. Em nên làm gì?', ['Từ chối', 'Chỉ gửi một phần mật khẩu để kiểm tra', 'Đổi mật khẩu rồi gửi mật khẩu cũ'], 0, 'Mật khẩu là thông tin riêng tư.'), ('Em nhận được tin nhắn thúc giục chuyển tiền vì có việc khẩn cấp. Em nên làm gì?', ['Chuyển một khoản nhỏ trước', 'Xác minh qua kênh khác', 'Nhắn hỏi lại ngay trên cùng tài khoản'], 1, 'Tin khẩn cấp cần được xác minh qua kênh khác.'), ('Bài học nhắc học sinh không chia sẻ mã OTP. Em nên hiểu thông tin này thế nào?', ['Đó là yêu cầu cung cấp mã OTP', 'Đó là lời nhắc bảo vệ tài khoản', 'Em phải gửi mã OTP cho giáo viên'], 1, 'Lời nhắc không chia sẻ mã OTP là nội dung giáo dục an toàn số.'), ('Một người tự nhận là giáo viên yêu cầu em nộp phí để nhận học bổng. Em nên làm gì?', ['Liên hệ trực tiếp với thầy cô hoặc nhà trường', 'Yêu cầu người gửi chụp ảnh thẻ giáo viên', 'Chờ người gửi xác nhận lại bằng tin nhắn'], 0, 'Hãy liên hệ trực tiếp với nhà trường.'), ('Nếu đã mở một đường liên kết đáng ngờ, em nên làm gì?', ['Thoát trang và tiếp tục sử dụng tài khoản bình thường', 'Ngừng thao tác và báo người lớn', 'Gửi liên kết cho bạn kiểm tra giúp'], 1, 'Ngừng thao tác và nhờ người lớn hỗ trợ.'), ('Khi phần mềm đưa ra mức cảnh báo thấp, em nên làm gì?', ['Có thể làm theo nếu nội dung không yêu cầu tiền', 'Vẫn phải kiểm tra người gửi', 'Có thể bỏ qua bước xác minh nếu tin ngắn'], 1, 'Cảnh báo thấp không có nghĩa là chắc chắn an toàn.'), ('Người lạ gửi đường liên kết để nhận quà miễn phí. Em nên làm gì?', ['Kiểm tra nguồn và hỏi người lớn', 'Nhập mật khẩu', 'Gửi mã OTP'], 0, 'Không đăng nhập qua đường dẫn chưa xác minh.'), ('Người lạ hỏi xin ảnh giấy tờ tùy thân của bố mẹ em. Em nên làm gì?', ['Gửi ngay', 'Từ chối và báo người lớn', 'Đăng lên nhóm'], 1, 'Giấy tờ tùy thân là thông tin riêng tư.'), ('Một người nhắn tin tự nhận là cô giáo nhưng xin chuyển tiền gấp?', ['Chuyển ngay', 'Hỏi trực tiếp cô qua kênh đã biết', 'Đưa mật khẩu'], 1, 'Tên và ảnh đại diện có thể bị giả mạo.'), ('Một người gửi tệp cài đặt ứng dụng không rõ nguồn gốc để xem điểm thi. Em nên làm gì?', ['Cài ngay', 'Gửi tiếp cho cả lớp', 'Không cài và hỏi giáo viên'], 2, 'Tệp cài đặt không rõ nguồn có thể gây hại.'), ('Tin nhắn bảo quét mã QR để đăng nhập nhận quà?', ['Quét và nhập mật khẩu', 'Không đăng nhập khi chưa xác minh', 'Nhờ bạn nhập hộ'], 1, 'Mã QR cũng có thể dẫn tới trang giả mạo.'), ('Tài khoản của bạn thân nhắn mượn tiền, nhưng cách nói chuyện khác thường. Em nên làm gì?', ['Gọi trực tiếp để xác minh', 'Chuyển ngay', 'Gửi mã OTP'], 0, 'Tài khoản người quen có thể bị chiếm quyền.'), ('Người lạ hỏi địa chỉ nhà và giờ em ở nhà một mình?', ['Trả lời đầy đủ', 'Gửi vị trí', 'Không cung cấp và báo người lớn'], 2, 'Không chia sẻ lịch sinh hoạt riêng tư.'), ('Một trang web yêu cầu tắt bảo mật điện thoại để nhận quà?', ['Làm theo', 'Không tắt bảo mật', 'Đưa điện thoại cho người lạ'], 1, 'Không tắt các tính năng bảo vệ theo lời người lạ.'), ('Tin nhắn báo tài khoản bị khóa và giục bấm đường liên kết trong 5 phút?', ['Bấm ngay', 'Nhập mật khẩu', 'Tự mở ứng dụng chính thức để kiểm tra'], 2, 'Kẻ xấu thường tạo áp lực thời gian.'), ('Người lạ đề nghị chia sẻ màn hình để giúp lấy lại tài khoản?', ['Từ chối và tìm hỗ trợ chính thức', 'Chia sẻ ngay', 'Gửi mã OTP'], 0, 'Chia sẻ màn hình có thể lộ dữ liệu.'), ('Có người đe dọa tung ảnh riêng tư nếu em không chuyển tiền?', ['Chuyển tiền', 'Lưu bằng chứng và báo người lớn', 'Gửi thêm ảnh'], 1, 'Không làm theo đe dọa, hãy tìm hỗ trợ.'), ('Em đọc được bài đăng nhắc mọi người không chia sẻ mã OTP. Em nên nhận xét thế nào?', ['Đây có thể là lời nhắc an toàn', 'Phải gửi mã OTP', 'Chắc chắn lừa đảo'], 0, 'Cần phân biệt lời cảnh báo với yêu cầu nguy hiểm.'), ('Một người tự nhận là nhân viên giao hàng yêu cầu em đọc mã xác thực vừa nhận được. Em nên làm gì?', ['Đọc mã ngay', 'Đăng mã lên nhóm', 'Không cung cấp mã và kiểm tra đơn hàng'], 2, 'Không tiết lộ mã đăng nhập hoặc xác thực.'), ('Em nhận lời mời làm nhiệm vụ trên mạng phải nạp tiền trước?', ['Nạp thử ít tiền', 'Không nạp tiền và hỏi người lớn', 'Vay tiền để nạp'], 1, 'Lời hứa kiếm tiền dễ dàng kèm yêu cầu nạp tiền là dấu hiệu đáng nghi.'), ('Nhóm lớp thông báo lịch kiểm tra qua kênh chính thức?', ['Có thể là thông báo học tập', 'Phải chuyển tiền', 'Phải gửi mật khẩu'], 0, 'Thông báo học tập thông thường vẫn nên đối chiếu lịch chính thức.'), ('Một người gửi đường liên kết bình chọn và yêu cầu đăng nhập mạng xã hội. Em nên làm gì?', ['Đăng nhập ngay', 'Gửi mật khẩu', 'Xác minh đường liên kết trước, không nhập mật khẩu ở trang lạ'], 2, 'Lời mời bình chọn có thể dẫn đến trang giả.'), ('Em nghi ngờ tài khoản của mình đã bị người khác chiếm quyền. Em nên làm gì?', ['Bỏ qua', 'Đổi mật khẩu qua ứng dụng chính thức và báo người lớn', 'Đưa mật khẩu cho người lạ'], 1, 'Hãy bảo vệ tài khoản qua kênh chính thức.'), ('Người lạ hứa tặng vật phẩm trong trò chơi nếu em gửi mật khẩu. Em nên làm gì?', ['Từ chối', 'Gửi mật khẩu', 'Đăng mật khẩu'], 0, 'Không đổi mật khẩu lấy quà.'), ('Tin nhắn báo tai nạn và giục mở đường liên kết để xem hình ảnh?', ['Mở đường liên kết ngay', 'Chia sẻ ngay', 'Gọi xác minh qua số quen thuộc, không bấm đường liên kết'], 2, 'Tin gây hoảng sợ có thể dụ bấm đường liên kết.'), ('Thầy cô nhắc không cung cấp thông tin cá nhân cho người lạ?', ['Làm ngược lại', 'Ghi nhớ và thực hiện', 'Gửi thông tin để xác nhận'], 1, 'Đây là lời nhắc bảo vệ thông tin.'), ('Một tài khoản bán hàng yêu cầu đặt cọc nhưng em chưa xác minh được người bán. Em nên làm gì?', ['Kiểm tra người bán và hỏi người lớn', 'Chuyển ngay', 'Gửi mã OTP'], 0, 'Xác minh trước khi giao dịch.'), ('Có tin nhắn nói em trúng thưởng nhưng phải nộp phí?', ['Nộp phí', 'Gửi giấy tờ', 'Không chuyển tiền, kiểm tra nguồn'], 2, 'Nộp phí trước để nhận thưởng là dấu hiệu đáng nghi.'), ('Em nhận được tin nhắn yêu cầu cài ứng dụng qua đường liên kết lạ. Em nên làm gì?', ['Cài ngay', 'Không cài, kiểm tra nguồn chính thức', 'Gửi cho bạn cài'], 1, 'Ứng dụng không rõ nguồn có thể chứa mã độc.'), ('Một tài khoản lạ yêu cầu cung cấp ngày sinh và địa chỉ để nhận học bổng?', ['Hỏi nhà trường qua kênh chính thức', 'Cung cấp ngay', 'Gửi ảnh giấy tờ'], 0, 'Cần xác minh chương trình học bổng trước khi chia sẻ dữ liệu.')]
with tab2:
    st.subheader('🎮 Thử thách – Nhận diện dấu hiệu lừa đảo')
    st.write('Trả lời 10 câu hỏi, xem lời giải thích và thử lại với bộ câu hỏi mới.')
    if 'quiz_round_33' not in st.session_state:
        st.session_state.quiz_round_33 = 0
    if 'quiz_items_33' not in st.session_state:
        st.session_state.quiz_items_33 = random.sample(range(len(QUIZ_BANK)), 10)
        st.session_state.quiz_result_33 = None
        st.session_state.quiz_option_orders_33 = {j: random.sample(range(3), 3) for j in range(10)}
    with st.form(f'quiz_form_33_{st.session_state.quiz_round_33}'):
        answers = []
        for j, idx in enumerate(st.session_state.quiz_items_33):
            question, options, correct, explanation = QUIZ_BANK[idx]
            order = st.session_state.quiz_option_orders_33[j]
            choice = st.radio(f'Câu {j+1}. {question}', [options[k] for k in order], index=None, key=f'quiz_answer_33_{st.session_state.quiz_round_33}_{j}')
            answers.append(choice)
        submitted = st.form_submit_button('✅ CHẤM ĐIỂM')
    if submitted:
        if any(a is None for a in answers):
            st.warning('Em hãy trả lời đủ 10 câu trước khi chấm điểm nhé!')
        else:
            st.session_state.quiz_result_33 = list(answers)
    if st.session_state.quiz_result_33 is not None:
        saved = st.session_state.quiz_result_33
        score = sum(saved[j] == QUIZ_BANK[idx][1][QUIZ_BANK[idx][2]] for j,idx in enumerate(st.session_state.quiz_items_33))
        st.success(f'🎉 Em trả lời đúng {score}/10 câu!')
        with st.expander('📖 Xem đáp án và giải thích từng câu', expanded=True):
            for j,idx in enumerate(st.session_state.quiz_items_33):
                question, options, correct, explanation = QUIZ_BANK[idx]
                ok = saved[j] == options[correct]
                st.write(f'{"✅" if ok else "❌"} **Câu {j+1} – Đáp án đúng:** {options[correct]}')
                st.write('**Vì sao?** ' + explanation)
        if st.button('🔄 THỬ THÁCH LẠI', key='quiz_again_33'):
            previous = set(st.session_state.quiz_items_33)
            candidates = [k for k in range(len(QUIZ_BANK)) if k not in previous]
            st.session_state.quiz_items_33 = random.sample(candidates, 10) if len(candidates) >= 10 else random.sample(range(len(QUIZ_BANK)), 10)
            st.session_state.quiz_round_33 += 1
            st.session_state.quiz_result_33 = None
            st.session_state.quiz_option_orders_33 = {j: random.sample(range(3), 3) for j in range(10)}
            st.rerun()

# Bộ kiểm thử minh họa được viết riêng, nhưng chưa đủ độc lập để công bố độ chính xác.
EXAMPLES = [
    ('Em được chọn nhận học bổng 5 triệu, hãy chuyển trước 200 nghìn phí xác nhận', 1),
    ('Nhận quà miễn phí nhưng cần đóng phí giao hàng trước', 1),
    ('Tài khoản của em bị khóa, bấm link để đăng nhập ngay', 1),
    ('Gửi mã OTP cho nhân viên để xác minh', 1),
    ('Nộp phí trước để nhận phần thưởng', 1),
    ('Cô giáo nhắc học sinh chuẩn bị bài kiểm tra', 0),
    ('Ngày mai lớp mình học tại phòng máy', 0),
    ('Bài học hôm nay hướng dẫn không chia sẻ OTP', 0),
    ('Nhóm chuẩn bị bài trình chiếu vào chiều thứ năm', 0),
    ('Hãy cảnh giác với tin nhắn yêu cầu chuyển tiền', 0),
]
# Trò chơi rèn kỹ năng, không dùng làm dữ liệu đánh giá mô hình AI.
DETECTIVE = [
    ('Bạn của em vừa gặp tai nạn! Hãy nhấn vào đường liên kết này để xem hình ảnh ngay.', 'Không đáng tin cậy', 'Tin nhắn lợi dụng thông tin về tai nạn để gây lo lắng, khiến người nhận vội mở đường liên kết không rõ nguồn gốc.', 'Không mở đường liên kết lạ. Hãy gọi trực tiếp cho bạn hoặc người thân của bạn để xác minh thông tin.'),
    ('Trong nhóm liên lạc chính thức của lớp, giáo viên nhắc học sinh mang sách Tin học vào ngày mai.', 'Có dấu hiệu đáng tin cậy', 'Đây là lời nhắc học tập thông thường từ kênh thông tin đã được xác nhận; không yêu cầu tiền hay thông tin bí mật.', 'Em có thể chuẩn bị sách vở và kiểm tra lại thời khóa biểu nếu cần.'),
    ('Em đã nhận được học bổng 5 triệu đồng! Hãy chuyển 200.000 đồng phí xác nhận ngay.', 'Không đáng tin cậy', 'Yêu cầu chuyển tiền trước để nhận học bổng là dấu hiệu đáng nghi.', 'Không chuyển tiền. Hãy hỏi trực tiếp giáo viên hoặc nhà trường để xác minh.'),
    ('Một số điện thoại lạ nhắn: Ngày mai trường em nghỉ học, hãy báo cho cả lớp nhé.', 'Cần xác minh thêm', 'Người gửi không rõ danh tính và chưa có thông báo chính thức từ nhà trường.', 'Hãy hỏi giáo viên hoặc kiểm tra kênh thông báo chính thức trước khi chia sẻ.'),
    ('Tài khoản của em sẽ bị khóa trong 5 phút! Hãy mở đường liên kết này và nhập mật khẩu ngay.', 'Không đáng tin cậy', 'Tin nhắn tạo áp lực về thời gian và yêu cầu nhập mật khẩu qua đường liên kết chưa xác minh.', 'Không mở đường liên kết. Hãy tự vào ứng dụng hoặc trang chính thức để kiểm tra.'),
    ('Bạn cùng nhóm học tập nhắc: Chiều thứ sáu, nhóm mình họp làm bài trình chiếu tại thư viện nhé.', 'Có dấu hiệu đáng tin cậy', 'Đây là lời hẹn học nhóm thông thường và không có yêu cầu cung cấp thông tin bí mật hay chuyển tiền.', 'Em có thể đối chiếu với lịch hoạt động của nhóm trước khi đến.'),
    ('Tôi là nhân viên hỗ trợ tài khoản. Em hãy đọc mã xác thực OTP vừa nhận được để tôi kiểm tra.', 'Không đáng tin cậy', 'Mã xác thực OTP là thông tin bảo mật, không được cung cấp cho người khác.', 'Không đọc hoặc gửi mã OTP. Hãy ngừng trao đổi và báo người lớn nếu tiếp tục bị thúc ép.'),
    ('Một tài khoản dùng ảnh đại diện giống cô giáo nhắn: Em hãy gửi số điện thoại của bố mẹ cho cô.', 'Cần xác minh thêm', 'Ảnh đại diện có thể bị giả mạo, nên chưa đủ căn cứ xác định người gửi là cô giáo.', 'Hãy gặp hoặc liên hệ trực tiếp với cô giáo qua kênh đã biết trước khi cung cấp thông tin.'),
    ('Cổng thông tin chính thức của trường thông báo: Tuần sau, nhà trường tổ chức hoạt động đọc sách tại thư viện.', 'Có dấu hiệu đáng tin cậy', 'Thông báo được đăng trên kênh chính thức của trường và không yêu cầu thông tin bí mật.', 'Em có thể xem thêm lịch hoạt động của trường để chuẩn bị tham gia.'),
    ('Tài khoản của một người bạn thân nhắn hỏi mượn tiền gấp, nhưng cách diễn đạt khác hẳn ngày thường.', 'Cần xác minh thêm', 'Tài khoản quen có thể bị người khác chiếm quyền sử dụng; chưa thể xác nhận người nhắn là bạn em.', 'Hãy gọi trực tiếp cho bạn qua số điện thoại đã biết trước khi quyết định.'),
]

# Bổ sung 90 tình huống độc lập theo ba nhóm nhận định; tổng 100 tình huống.
DETECTIVE.extend([('Em trúng học bổng nhưng phải đóng phí xác nhận ngay hôm nay.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Nhận iPhone miễn phí, chỉ cần nộp phí vận chuyển trước.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Hãy đọc mã OTP để tôi mở khóa tài khoản giúp em.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Tài khoản sắp bị khóa, nhập mật khẩu qua link đính kèm.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Cần nạp 100 nghìn để nhận thưởng 2 triệu từ trò chơi.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Gửi ảnh căn cước của bố mẹ để nhận quà từ người lạ.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Tôi có thể tăng lượt theo dõi nếu em gửi mật khẩu Facebook.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Nhấp vào đường dẫn để xem ảnh tai nạn của bạn em.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Người quen qua mạng rủ em đi xa và dặn giấu gia đình.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Tôi sẽ tặng em điện thoại nếu em đến nhà tôi một mình.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Hãy gửi ảnh riêng tư, nếu không tôi sẽ đăng tin xấu về em.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Một người đe dọa tung ảnh nhạy cảm nếu em không chuyển tiền.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Tải file APK lạ để xem điểm thi trước các bạn.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Quét mã QR và đăng nhập tài khoản để nhận phần thưởng.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Tôi là nhân viên ngân hàng, gửi mật khẩu để xác minh.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Bấm link này để nhận kim cương miễn phí trong game.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Hãy chuyển khoản đặt cọc để được nhận công việc online.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Một người lạ bảo em bật chia sẻ màn hình để sửa tài khoản.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Người quen qua game yêu cầu em giữ bí mật cuộc gặp.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Bạn sẽ bị phạt nếu không gửi mã xác thực trong 5 phút.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Đừng báo giáo viên, nếu mách tôi sẽ đánh em sau giờ học.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Một tài khoản ép em quay video riêng tư để chứng minh tình bạn.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Tôi biết địa chỉ nhà em, hãy ra gặp tôi ngay và đừng nói ai.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Người lạ nhắn muốn biết giờ em ở nhà một mình.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Chuyển 50 nghìn để được vào nhóm xem đề thi bí mật.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Một người tự xưng công an yêu cầu em chuyển tiền để tránh bị bắt.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Hãy cho tôi điều khiển máy tính từ xa để nhận quà.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Tài khoản giả giáo viên yêu cầu nộp tiền vào ví cá nhân ngay.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Bạn bè rủ em thực hiện thử thách nguy hiểm rồi quay video.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Người lạ hứa trả tiền nếu em cung cấp danh sách số điện thoại bạn học.', 'Không đáng tin cậy', 'Tin nhắn có dấu hiệu dụ dỗ, lừa đảo hoặc yêu cầu nguy hiểm.', 'Không làm theo; giữ bằng chứng nếu cần và nhờ người lớn đáng tin cậy hỗ trợ.'), ('Một số lạ nhắn trường nghỉ học ngày mai, bảo em báo cả lớp.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Tài khoản giống cô giáo gửi lịch kiểm tra mới nhưng chưa thấy thông báo lớp.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Bạn thân nhắn mượn tiền nhưng cách nói chuyện khác thường.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một người tự nhận là anh họ xin số điện thoại của bố mẹ.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Có tin nhắn nói lịch thi đổi giờ nhưng không nêu nguồn.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một trang mạng đăng thông báo học bổng mà không có tên đơn vị cấp.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Tài khoản dùng ảnh cô giáo hỏi em ngày sinh và địa chỉ.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một người nhắn bảo đã chuyển nhầm tiền và yêu cầu em hoàn trả gấp.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Nhóm chat lan truyền ảnh nói trường sẽ đóng cửa một tuần.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một tài khoản lạ mời em tham gia câu lạc bộ của trường.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Người bán hàng nói có giảm giá đặc biệt nhưng chưa có thông tin cửa hàng.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Em nhận mã QR thanh toán từ người chưa từng giao dịch.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một đường dẫn tự nhận là cổng điểm học sinh nhưng địa chỉ lạ.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Bạn chuyển tiếp video được cho là phát biểu của hiệu trưởng.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một số lạ báo em có bưu kiện cần xác nhận địa chỉ.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Người nhắn nói là nhân viên nhà mạng nhưng không có kênh xác minh.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Có tin nhắn báo nhóm lớp chuyển sang ứng dụng khác.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một trang mạng nói tất cả học sinh sẽ được tặng máy tính.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Tài khoản mới lập nhắn mời em tham dự cuộc thi vẽ.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một người nói là phụ huynh bạn em và nhờ gửi vị trí của bạn.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Tin nhắn cho biết có lịch tiêm chủng mới nhưng không rõ đơn vị.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một bạn gửi đường dẫn bình chọn nhưng cần đăng nhập tài khoản.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một tài khoản thông báo thu tiền hoạt động ngoại khóa không có xác nhận.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Tin nhắn báo em đã đăng ký một khóa học dù em không nhớ.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một người tự nhận là giáo viên mới và xin danh sách lớp.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Trang lạ nói ngày mai được nghỉ do thời tiết xấu.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Bạn gửi ảnh chụp thông báo nhà trường nhưng thiếu ngày tháng.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Có tin đồn thầy cô sẽ đổi lịch thi cuối kỳ.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Một người nhắn em được chọn làm đại sứ học sinh nhưng chưa rõ tổ chức.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Người lạ hỏi em có muốn tham gia nhóm học trực tuyến không.', 'Cần xác minh thêm', 'Chưa đủ căn cứ xác định thông tin hoặc danh tính người gửi.', 'Kiểm tra qua kênh chính thức hoặc liên hệ trực tiếp người quen; không cung cấp thông tin riêng.'), ('Giáo viên nhắc lớp mang sách Tin học ngày mai trong nhóm chính thức.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Bạn cùng lớp hẹn học nhóm tại thư viện trường vào chiều thứ sáu.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Nhà trường đăng lịch sinh hoạt dưới cờ trên cổng thông tin chính thức.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Cô giáo nhắc không chia sẻ mã OTP cho người khác.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Nhóm trực nhật nhắc đến sớm để dọn lớp.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Giáo viên gửi bài tập về nhà trên hệ thống học tập của trường.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Bạn cùng bàn hỏi em đã làm xong bài thực hành chưa.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Lớp trưởng nhắc cả lớp mang giấy kiểm tra vào tiết sau.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Thầy giáo hướng dẫn cách nhận biết tin nhắn giả mạo.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Thư viện trường thông báo lịch hoạt động đọc sách trên kênh chính thức.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Bạn học nhắn chúc em làm bài thi tốt.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Cô giáo nhắc các em kiểm tra lại thời khóa biểu chính thức.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Nhóm học tập thống nhất phân chia nội dung thuyết trình.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Giáo viên dặn không nhấp vào liên kết không rõ nguồn gốc.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Bạn cùng lớp mời em tham gia hoạt động STEM tại trường.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Câu lạc bộ trường thông báo lịch sinh hoạt trên bảng tin chính thức.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Thầy giáo hướng dẫn sử dụng mật khẩu mạnh và xác thực hai bước.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Cô chủ nhiệm thông báo lịch họp phụ huynh qua nhóm đã xác minh.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Bạn nhắc em đem áo thể dục theo thời khóa biểu.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Nhà trường đăng thông báo ngày hội ngoại ngữ trên website chính thức.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Bạn hỏi em có muốn cùng ôn tập ở phòng học nhóm không.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Cô giáo nhắc nộp bài tập trước hạn qua Classroom của lớp.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Thầy hướng dẫn học sinh không gửi ảnh giấy tờ tùy thân lên mạng.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Nhóm bạn thống nhất giờ tập văn nghệ ở sân trường.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Giáo viên nhắc học sinh bảo vệ quyền riêng tư khi đăng ảnh.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Cô giáo công bố lịch kiểm tra trong tiết học trực tiếp.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Bạn cùng lớp nhắc chuẩn bị USB để làm bài thực hành.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Nhà trường nhắc học sinh đội mũ bảo hiểm khi đến trường.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Giáo viên hướng dẫn báo người lớn khi bị bắt nạt trực tuyến.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.'), ('Nhóm lớp nhắc các bạn chuẩn bị tài liệu cho buổi thảo luận.', 'Có dấu hiệu đáng tin cậy', 'Nội dung học tập hoặc sinh hoạt thông thường từ kênh quen thuộc, không có dấu hiệu rủi ro rõ ràng.', 'Có thể tiếp nhận thông tin; vẫn đối chiếu lịch và giữ an toàn thông tin cá nhân.')])

DETECTIVE_CHOICES = ['Có dấu hiệu đáng tin cậy', 'Không đáng tin cậy', 'Cần xác minh thêm']

with tab3:
    st.subheader('🔍 Giải mã tin nhắn – Em có nhận ra tin nhắn đáng nghi?')
    st.write('Ngân hàng **100 tin nhắn**; mỗi lượt chọn ngẫu nhiên **10 tin nhắn** để em nhận định và xem hướng dẫn.')
    if 'detective_round_33' not in st.session_state:
        st.session_state.detective_order_33 = random.sample(range(len(DETECTIVE)), 10)
        st.session_state.detective_round_33 = 0
        st.session_state.detective_score_33 = 0
        st.session_state.detective_feedback_33 = None
    i = st.session_state.detective_round_33
    if i < len(st.session_state.detective_order_33):
        msg, truth, why, action = DETECTIVE[st.session_state.detective_order_33[i]]
        st.progress(i / len(st.session_state.detective_order_33))
        st.write(f'**Tình huống {i+1}/{len(st.session_state.detective_order_33)}**')
        st.info(msg)
        if st.session_state.detective_feedback_33 is None:
            with st.form(f'detective_form_33_{i}'):
                pick = st.radio('Em nhận xét thế nào?', DETECTIVE_CHOICES, index=None, key=f'detective_pick_33_{i}')
                check = st.form_submit_button('🔎 KIỂM TRA')
            if check:
                if pick is None:
                    st.warning('Em hãy chọn một câu trả lời nhé!')
                else:
                    correct = pick == truth
                    if correct:
                        st.session_state.detective_score_33 += 1
                    st.session_state.detective_feedback_33 = correct
                    st.rerun()
        else:
            if st.session_state.detective_feedback_33:
                st.success('🎉 Chính xác! Em đã nhận diện đúng.')
            else:
                st.warning('Em chưa chọn đúng. Cùng xem lý do và ghi nhớ cách xử lý nhé!')
            st.write('**Đáp án đúng:** ' + truth)
            st.write('**Vì sao?** ' + why)
            st.write('**Em nên làm gì?** ' + action)
            if st.button('➡️ TÌNH HUỐNG TIẾP THEO', key=f'detective_next_33_{i}'):
                st.session_state.detective_round_33 += 1
                st.session_state.detective_feedback_33 = None
                st.rerun()
    else:
        st.success(f'🏆 Em đã hoàn thành! Kết quả: {st.session_state.detective_score_33}/{len(st.session_state.detective_order_33)} tình huống.')
        st.write('Hãy luôn kiểm tra nguồn gửi trước khi làm theo tin nhắn.')
        if st.button('🔄 CHƠI LẠI', key='detective_replay_33'):
            st.session_state.detective_round_33 = 0
            st.session_state.detective_order_33 = random.sample(range(len(DETECTIVE)), 10)
            st.session_state.detective_score_33 = 0
            st.session_state.detective_feedback_33 = None
            st.rerun()


# Tình huống giáo dục giả lập, không dùng để đánh giá độ chính xác của mô hình.
SITUATIONS = [('Học bổng bất ngờ', 'Em nhận được tin nhắn báo trúng học bổng, nhưng người gửi yêu cầu nộp phí xác nhận ngay.', ['Nộp tiền để giữ suất học bổng', 'Liên hệ trực tiếp với giáo viên hoặc nhà trường để xác minh', 'Gửi thông tin tài khoản cho người nhắn'], 1, 'Học bổng cần được xác minh qua nhà trường. Không chuyển tiền hoặc cung cấp thông tin riêng tư chỉ vì một tin nhắn.'), ('Giả danh giáo viên', 'Một tài khoản lạ dùng ảnh đại diện giống giáo viên của em và yêu cầu chuyển tiền gấp.', ['Gọi điện hoặc gặp trực tiếp giáo viên để kiểm tra', 'Chuyển tiền ngay vì nhận ra ảnh đại diện', 'Gửi mã xác thực OTP cho tài khoản đó'], 0, 'Ảnh đại diện và tên tài khoản có thể bị giả mạo. Hãy xác minh bằng cách liên hệ trực tiếp với giáo viên.'), ('Đường liên kết lạ', 'Em nhận được tin nhắn yêu cầu mở một đường liên kết để tránh bị khóa tài khoản.', ['Mở đường liên kết và nhập mật khẩu', 'Chuyển đường liên kết cho bạn thử', 'Tự mở ứng dụng chính thức để kiểm tra tài khoản'], 2, 'Không đăng nhập qua đường liên kết chưa rõ nguồn gốc. Hãy kiểm tra tài khoản trong ứng dụng hoặc trang web chính thức.'), ('Trúng thưởng điện thoại', 'Người lạ thông báo em trúng điện thoại, nhưng phải trả phí vận chuyển trước.', ['Chuyển phí vận chuyển ngay', 'Không chuyển tiền và kiểm tra thông tin chương trình', 'Gửi ảnh giấy tờ tùy thân để nhận thưởng'], 1, 'Yêu cầu trả tiền trước để nhận thưởng là dấu hiệu đáng nghi. Hãy trao đổi với người lớn trước khi quyết định.'), ('Bị đe dọa trên mạng', 'Một người đe dọa phát tán hình ảnh riêng tư của em nếu em không chuyển tiền.', ['Chuyển tiền để người đó giữ im lặng', 'Lưu bằng chứng và báo ngay cho người lớn đáng tin cậy', 'Gửi thêm hình ảnh để thương lượng'], 1, 'Không chuyển tiền và không gửi thêm hình ảnh. Hãy lưu tin nhắn làm bằng chứng và tìm người lớn hỗ trợ ngay.'), ('Yêu cầu mã xác thực OTP', 'Một người tự nhận là nhân viên hỗ trợ tài khoản và yêu cầu em đọc mã xác thực OTP vừa nhận được.', ['Đọc mã OTP vì người đó tự nhận là nhân viên', 'Không cung cấp mã OTP và báo người lớn nếu bị thúc ép', 'Đăng mã OTP lên nhóm lớp để hỏi ý kiến'], 1, 'Mã xác thực OTP có thể dùng để đăng nhập hoặc thực hiện giao dịch. Không cung cấp mã cho người khác.'), ('Quà tặng trong trò chơi', 'Một người hứa tặng vật phẩm trong trò chơi nếu em cung cấp mật khẩu tài khoản.', ['Gửi mật khẩu để nhận quà', 'Đổi mật khẩu thành tên mình cho dễ nhớ', 'Từ chối và giữ bí mật mật khẩu'], 2, 'Không cung cấp mật khẩu để đổi lấy quà tặng. Hãy sử dụng mật khẩu mạnh và giữ bí mật thông tin đăng nhập.'), ('Bạn thân nhắn mượn tiền', 'Tài khoản của một người bạn thân nhắn hỏi mượn tiền gấp, nhưng cách diễn đạt khác hẳn ngày thường.', ['Gọi trực tiếp cho bạn qua số đã biết để xác minh', 'Chuyển tiền ngay vì đó là tài khoản của bạn', 'Gửi mã OTP để xác nhận giúp bạn'], 0, 'Tài khoản của người quen cũng có thể bị chiếm quyền. Hãy xác minh qua cuộc gọi hoặc gặp trực tiếp.'), ('Mã QR nhận quà', 'Em được yêu cầu quét một mã QR lạ rồi đăng nhập tài khoản để nhận quà.', ['Quét mã và nhập mật khẩu ngay', 'Không đăng nhập khi chưa xác minh nguồn của mã QR', 'Nhờ bạn đăng nhập hộ bằng tài khoản của mình'], 1, 'Mã QR có thể dẫn đến trang giả mạo. Hãy kiểm tra nguồn trước khi quét và không nhập mật khẩu vào trang lạ.'), ('Tệp cài đặt không rõ nguồn', 'Người lạ gửi tệp cài đặt ứng dụng và bảo em cài vào điện thoại để xem điểm kiểm tra.', ['Cài ngay để xem điểm', 'Chuyển tệp cho cả lớp cùng cài', 'Không cài và hỏi giáo viên về cách xem điểm chính thức'], 2, 'Tệp cài đặt từ nguồn lạ có thể gây hại cho thiết bị. Chỉ cài ứng dụng từ nguồn đáng tin cậy.'), ('Đe dọa khóa tài khoản', 'Tin nhắn thông báo tài khoản sẽ bị khóa sau 10 phút nếu em không đăng nhập qua đường liên kết được gửi kèm.', ['Tự mở ứng dụng chính thức để kiểm tra', 'Mở ngay đường liên kết vì sợ bị khóa', 'Nhập mật khẩu vào trang được gửi trong tin nhắn'], 0, 'Tin nhắn tạo áp lực thời gian có thể nhằm khiến em hành động vội vàng. Hãy kiểm tra bằng kênh chính thức.'), ('Bình chọn cuộc thi', 'Một tài khoản gửi đường liên kết và nhờ em đăng nhập mạng xã hội để bình chọn cho bạn.', ['Đăng nhập ngay qua đường liên kết', 'Xác minh nguồn và không nhập mật khẩu vào trang lạ', 'Gửi mật khẩu cho người nhờ bình chọn'], 1, 'Trang bình chọn giả mạo có thể đánh cắp thông tin đăng nhập. Hãy kiểm tra địa chỉ trang và nguồn gửi.'), ('Giả danh người thân', 'Một số điện thoại lạ tự nhận là người thân và yêu cầu em gửi ảnh giấy tờ của gia đình.', ['Gửi ngay vì người đó biết tên em', 'Đăng ảnh giấy tờ lên nhóm để hỏi mọi người', 'Gọi cho người thân qua số điện thoại đã lưu để xác minh'], 2, 'Không gửi giấy tờ tùy thân cho người chưa xác minh danh tính. Hãy liên hệ người thân bằng kênh quen thuộc.'), ('Lời mời kiếm tiền trên mạng', 'Một người rủ em làm nhiệm vụ trên mạng, nhưng yêu cầu nạp tiền trước mới được nhận thưởng.', ['Không nạp tiền và trao đổi với người lớn', 'Nạp một ít tiền để thử', 'Mượn tiền bạn để tham gia'], 0, 'Lời hứa kiếm tiền dễ dàng kèm yêu cầu nạp tiền trước là dấu hiệu đáng cảnh giác.'), ('Yêu cầu chia sẻ màn hình', 'Một người lạ đề nghị em chia sẻ màn hình điện thoại để giúp khôi phục tài khoản.', ['Cho phép người lạ điều khiển điện thoại từ xa', 'Từ chối và tìm kênh hỗ trợ chính thức', 'Gửi mật khẩu để người đó kiểm tra'], 1, 'Chia sẻ màn hình có thể làm lộ mã xác thực và dữ liệu riêng tư. Hãy nhờ người lớn hoặc bộ phận hỗ trợ chính thức.'), ('Thông báo họp phụ huynh', 'Nhóm lớp đăng thông báo họp phụ huynh và không yêu cầu chuyển tiền hay cung cấp mật khẩu.', ['Xóa ngay vì chắc chắn là lừa đảo', 'Gửi mã OTP để xác nhận đã đọc', 'Đối chiếu với thông báo từ giáo viên chủ nhiệm'], 2, 'Đây có thể là thông báo học tập bình thường. Em nên kiểm tra lại qua kênh liên lạc chính thức của lớp.'), ('Lời nhắc phòng tránh lừa đảo', 'Giáo viên nhắc học sinh không chia sẻ mã OTP và không mở đường liên kết lạ.', ['Đọc và ghi nhớ lời nhắc an toàn', 'Gửi mã OTP để xác nhận đã đọc', 'Chuyển tiền để chứng minh đã hiểu'], 0, 'Đây là nội dung giáo dục về an toàn số, không phải yêu cầu cung cấp thông tin bí mật.'), ('Tài khoản mạng xã hội bị chiếm', 'Em phát hiện tài khoản mạng xã hội của mình tự gửi những tin nhắn lạ cho bạn bè.', ['Bỏ qua vì nghĩ không có gì nghiêm trọng', 'Đổi mật khẩu qua ứng dụng chính thức và báo người lớn', 'Gửi mật khẩu cho người lạ để nhờ sửa'], 1, 'Hãy đổi mật khẩu, kiểm tra phiên đăng nhập và cảnh báo bạn bè. Nhờ người lớn hỗ trợ nếu cần.'), ('Người lạ hỏi địa chỉ', 'Một người mới quen trên mạng hỏi địa chỉ nhà và thời gian em ở nhà một mình.', ['Gửi vị trí hiện tại cho người đó', 'Chụp ảnh nhà và gửi ngay', 'Không chia sẻ thông tin riêng tư và báo người lớn nếu bị ép'], 2, 'Địa chỉ và lịch sinh hoạt là thông tin riêng tư. Không chia sẻ cho người lạ trên mạng.'), ('Mua hàng trên mạng', 'Một tài khoản bán hàng yêu cầu em chuyển tiền đặt cọc trước khi cho xem thông tin sản phẩm.', ['Kiểm tra người bán và hỏi người lớn trước khi giao dịch', 'Chuyển tiền đặt cọc ngay', 'Gửi mật khẩu tài khoản ngân hàng'], 0, 'Hãy xác minh người bán, sản phẩm và điều kiện giao dịch trước khi chuyển tiền.')]

with tab3:
    st.divider()
    st.subheader('🎭 Em sẽ làm gì? – Luyện cách xử lý tình huống')
    st.write('Có **20 tình huống an toàn trên mạng** để em luyện cách xử lý an toàn.')
    names = [item[0] for item in SITUATIONS]
    selected = st.selectbox('Chọn tình huống', names, key='situation_name_33')
    idx = names.index(selected)
    name, scenario, choices, correct, lesson = SITUATIONS[idx]
    st.info(scenario)
    with st.form(key=f'situation_form_33_{idx}'):
        answer = st.radio('Em sẽ làm gì?', choices, index=None, key=f'situation_answer_33_{idx}')
        checked = st.form_submit_button('KIỂM TRA CÁCH XỬ LÝ')
    if checked:
        if answer is None:
            st.warning('Em hãy chọn một cách xử lý trước nhé.')
        elif answer == choices[correct]:
            st.success('🎉 Chính xác! Em đã chọn cách xử lý an toàn.')
            st.write('**Lời khuyên:** ' + lesson)
        else:
            st.warning('⚠️ Cách xử lý này có thể khiến em gặp rủi ro.')
            st.write('**Cách xử lý phù hợp:** ' + choices[correct])
            st.write('**Lời khuyên:** ' + lesson)


# Tư vấn theo quy tắc: không dùng mô hình phân loại tin lừa đảo để chẩn đoán
# tình huống bạo lực, sức khỏe tâm thần hoặc nguy hiểm cá nhân.
def advise_student_situation(message):
    t = normalize(message)
    # Gom khoảng trắng để nhận diện cả khi học sinh xuống dòng hoặc gõ nhiều dấu cách.
    t = ' '.join(t.split())
    def contains(*phrases):
        return any(p in t for p in phrases)

    # 4.7: Phân tích nhiều tín hiệu nguy hiểm trước các nhánh cảnh báo chung.
    # Tránh kết luận nguy hiểm khi học sinh chỉ thuật lại lời khuyên phòng tránh.
    protective = contains('co giao dan', 'thay giao dan', 'me dan', 'bo dan',
        'khong nen', 'tuyet doi khong', 'khong duoc cung cap', 'dung bao gio',
        'canh bao rang', 'huong dan phong tranh')
    credential = contains('mat khau', 'otp', 'ma xac thuc', 'ma dang nhap')
    credential_request = contains('bao em dua', 'bao em gui', 'bao em doc',
        'yeu cau em', 'doi em', 'xin em', 'dua mat khau', 'dua ma',
        'gui mat khau', 'gui ma', 'cung cap mat khau', 'cung cap ma',
        'nhap mat khau', 'nhap ma otp', 'nhan ma otp', 'cho em dua',
        'bao em cung cap', 'doi lay', 'de duoc tang', 'de nhan qua')
    credential_bait = contains('kim cuong', 'mien phi', 'tang qua', 'tang vat pham',
        'trung thuong', 'doi qua', 'nhan qua', 'nhan thuong', 'tang 500',
        'game', 'nap the', 'tang skin')
    if credential and credential_request and not protective:
        return ('NGUY CƠ CAO', 'Có dấu hiệu yêu cầu tiết lộ mật khẩu hoặc mã OTP. Đây là thông tin bảo mật không được cung cấp để đổi lấy quà hay vật phẩm.', [
            'Không gửi mật khẩu, OTP hoặc mã xác thực cho bất kỳ ai, kể cả người tự nhận là hỗ trợ trò chơi.',
            'Ngừng trao đổi và báo cáo hoặc chặn tài khoản yêu cầu; nhờ cha mẹ hoặc thầy cô hỗ trợ.',
            'Nếu đã cung cấp thông tin, đổi mật khẩu ngay trên ứng dụng chính thức, đăng xuất phiên lạ và nhờ người lớn hỗ trợ bảo vệ tài khoản.'
        ])
    # Trường hợp rủ gặp người quen qua mạng, yêu cầu giữ bí mật với gia đình.
    online_acquaintance = contains('quen mot anh tren mang', 'quen 1 anh tren mang',
        'quen anh tren mang', 'quen mot ban trai tren mang', 'quen ban trai tren mang',
        'quen mot nguoi tren mang', 'quen nguoi tren mang', 'quen qua mang',
        'quen tren mang', 'tren mang', 'qua mang', 'qua game', 'online')
    meeting = contains('ru hai dua', 'ru ca hai', 'ru chung em', 'ru bon em',
        'ru em', 'ru di', 'di choi', 'di gap', 'hen gap', 'gap mat',
        'di xa', 'den nha', 'toi nha', 'hen em', 'ru toi')
    hidden = contains('dung cho bo me biet', 'dung cho cha me biet',
        'dung cho gia dinh biet', 'khong cho bo me biet', 'khong cho gia dinh biet',
        'khong noi voi bo me', 'khong noi voi cha me', 'khong ke voi bo me',
        'dung noi voi bo me', 'dung ke voi bo me', 'giau bo me', 'giau gia dinh',
        'khong duoc ke', 'giu bi mat', 'bi mat voi gia dinh',
        'bao dung cho', 'bao khong duoc noi', 'bao giu kin')
    if online_acquaintance and meeting and hidden and not protective:
        return ('NGUY CƠ CAO', 'Có dấu hiệu dụ dỗ gặp người quen qua mạng kèm yêu cầu giấu gia đình; không thể xác minh danh tính và mức độ an toàn của cuộc gặp.', [
            'Không đi gặp, không đi xa cùng người này và không giữ bí mật với gia đình.',
            'Báo ngay cho cha mẹ, thầy cô hoặc người lớn đáng tin cậy; cho họ xem lời nhắn.',
            'Không cung cấp vị trí, địa chỉ, lịch trình hay thông tin cá nhân; nếu bị đe dọa, hãy tìm hỗ trợ khẩn cấp.'
        ])
    # Ưu tiên phát hiện các dấu hiệu khẩn cấp trước mọi nhóm khác.
    if contains('tu tu', 'tu sat', 'muon chet', 'ket lieu', 'tu lam dau', 'lam hai ban than', 'cat tay'):
        return ('CẦN HỖ TRỢ NGAY', 'Em đang chia sẻ điều rất đáng quan tâm. Em không cần tự giải quyết một mình.', [
            'Hãy đến ngay với cha mẹ, thầy cô hoặc người lớn đáng tin cậy và nói rõ em đang cần được giúp đỡ.',
            'Nếu em có nguy cơ làm hại bản thân ngay lúc này, đừng ở một mình; hãy nhờ người lớn đưa đến cơ sở y tế hoặc gọi cấp cứu 115.',
            'Nếu khó nói thành lời, em có thể cho người lớn xem nội dung em vừa viết.'
        ])
    if contains('bi xam hai', 'xam hai tinh duc', 'cuong hiep', 'ep quan he', 'so mo', 'dung cham vung kin'):
        return ('CẦN HỖ TRỢ NGAY', 'Đây có thể là tình huống xâm hại nghiêm trọng.', [
            'Rời khỏi người hoặc nơi khiến em không an toàn nếu có thể.',
            'Báo ngay cho người lớn đáng tin cậy không liên quan đến hành vi xâm hại; em có thể gọi Tổng đài bảo vệ trẻ em 111.',
            'Nếu đang nguy hiểm trực tiếp, hãy nhờ người lớn gọi 113 hoặc đến nơi có người hỗ trợ. Không tự đối chất.'
        ])
    if contains('co giao dan', 'thay giao dan', 'bo me dan', 'me dan', 'thay co nhac') and contains('khong duoc cung cap', 'tuyet doi khong', 'khong nen dua', 'khong chia se'):
        return ('TÌNH HUỐNG BÌNH THƯỜNG', 'Đây là lời nhắc phòng tránh rủi ro, không phải yêu cầu cung cấp thông tin bảo mật.', [
            'Tiếp tục giữ kín mật khẩu và mã OTP.', 'Chỉ sử dụng các kênh chính thức khi cần hỗ trợ tài khoản.'])
    # 4.5: Dấu hiệu bổ sung từ ngân hàng tình huống THCS.
    # Các điều kiện có tính kết hợp để giảm cảnh báo nhầm.
    if contains('ep len xe', 'bi ep len xe', 'keu cuu', 'cau cuu') and contains('xe', 'bat', 'theo', 'cuu'):
        return ('CẦN HỖ TRỢ NGAY', 'Có dấu hiệu nguy hiểm trực tiếp, cần hỗ trợ khẩn cấp.', [
            'Báo ngay người lớn ở gần và gọi công an 113 nếu có nguy hiểm trực tiếp.',
            'Cung cấp vị trí và thông tin đang có; không tự đi giải cứu hay đối đầu.',
            'Trẻ em có thể liên hệ tổng đài 111 để được hỗ trợ.'])
    if contains('ghep mat', 'ghep anh', 'deepfake', 'anh gia') and contains('nhay cam', 'khoa than', 'phat tan', 'dang len mang', 'dang facebook'):
        return ('NGUY CƠ CAO', 'Có dấu hiệu tạo hoặc phát tán hình ảnh xâm phạm quyền riêng tư.', [
            'Không chia sẻ hoặc phát tán hình ảnh.', 'Giữ lại liên kết, tài khoản và tin nhắn làm bằng chứng, tránh lưu hoặc gửi lại ảnh nhạy cảm.',
            'Báo ngay người lớn đáng tin cậy để được hỗ trợ báo cáo nội dung và bảo vệ bản thân.'])
    if contains('thu thach nguy hiem', 'thu thach tren mang', 'quay clip lam theo'):
        return ('NGUY CƠ CAO', 'Lời rủ thực hiện thử thách nguy hiểm có thể gây tổn hại.', [
            'Không làm theo hoặc rủ bạn bè tham gia.', 'Báo người lớn và báo cáo nội dung trên nền tảng.'])
    if contains('nguoi la', 'nguoi moi quen', 'nguoi tren mang', 'ban tren mang') and contains('dia chi nha', 'o nha mot minh', 'gio o nha mot minh'):
        return ('NGUY CƠ CAO', 'Người chưa được xác minh hỏi thông tin vị trí hoặc lịch sinh hoạt riêng tư.', [
            'Không cung cấp địa chỉ, vị trí hoặc thời gian ở nhà một mình.', 'Ngừng trao đổi và báo cha mẹ hoặc thầy cô đáng tin cậy.'])
    if contains('nguoi la', 'nguoi moi quen', 'tren mang', 'qua game') and contains('mat khau', 'ma otp', 'ma xac thuc') and contains('gui', 'dua', 'doc', 'cung cap', 'xin'):
        return ('NGUY CƠ CAO', 'Có dấu hiệu yêu cầu tiết lộ thông tin bảo mật.', [
            'Không cung cấp mật khẩu hoặc mã xác thực.', 'Ngừng trao đổi, báo người lớn và dùng kênh chính thức để kiểm tra tài khoản.'])
    if contains('ban cung lop', 'ban trong lop', 'nhom lop') and contains('thu vien', 'hoc nhom') and not contains('giau bo me', 'giau gia dinh', 'khong cho gia dinh biet', 'khong cho bo me biet', 'khong cho cha me biet', 'khong duoc ke'):
        return ('TÌNH HUỐNG BÌNH THƯỜNG', 'Lời mời học nhóm thông thường, chưa thấy dấu hiệu nguy hiểm rõ ràng.', [
            'Có thể tham gia nếu em thấy thoải mái, biết rõ địa điểm và lịch trình.', 'Thông báo cho gia đình và tuân thủ quy định nhà trường.'])
    # Nhận diện dụ dỗ trực tuyến: kết hợp ngữ cảnh, không chỉ một từ khóa.
    online_person = contains('qua game', 'qua tro choi', 'choi game', 'tren game',
        'quen tren mang', 'quen qua mang', 'nguoi quen tren mang', 'nguoi quen qua mang',
        'nguoi la tren mang', 'ban tren mang', 'ban qua mang', 'nguoi la',
        'chua gap ngoai doi', 'chua tung gap', 'moi quen online', 'quen 1 ban tren mang', 'quen mot ban tren mang', 'quen ban trai tren mang', 'quen qua facebook', 'quen qua tiktok')
    meet_request = contains('ru em den', 'ru em toi', 'ru em di', 'ru toi den',
        'hen gap', 'gap mat', 'den nha', 'toi nha', 'qua nha', 'di choi',
        'di theo', 'ru den nha', 'ru toi nha', 'moi den nha', 'moi toi nha',
        'ru em qua', 'ru em sang', 'ru em di gap', 'ru di gap', 'di gap cung', 'di gap voi', 'di gap nguoi', 'ru di gap')
    secrecy = contains('giau bo me', 'giau cha me', 'giau gia dinh', 'khong duoc ke',
        'khong duoc noi', 'khong duoc bao', 'dung noi voi', 'dung ke voi',
        'khong cho bo me biet', 'khong cho cha me biet', 'khong cho ai biet', 'khong cho gia dinh biet', 'khong cho nha biet', 'dung cho gia dinh biet',
        'giu bi mat', 'khong de bo me biet', 'khong de cha me biet')
    gifts = contains('tang dien thoai', 'tang qua', 'cho dien thoai', 'cho qua',
        'hua tang', 'se tang', 'dien thoai moi', 'qua dat tien', 'nap the',
        'vat pham game', 'tang iphone')
    if (online_person and meet_request and (secrecy or gifts)) or (meet_request and secrecy and gifts) or (meet_request and secrecy and contains('ban trai', 'ban gai', 'tren mang', 'online')):
        return ('NGUY CƠ CAO',
            'Có dấu hiệu dụ dỗ gặp mặt không an toàn: người quen qua mạng hoặc lời hứa tặng quà đi kèm yêu cầu giữ bí mật. Chưa thể xác minh danh tính hay ý định người đó.', [
            'Không đến điểm hẹn, không đi theo hoặc gặp riêng người này, kể cả khi họ hứa tặng quà.',
            'Nói ngay với cha mẹ, giáo viên chủ nhiệm hoặc người lớn đáng tin cậy; cho họ xem tin nhắn để cùng xử lý.',
            'Không gửi địa chỉ, vị trí, lịch học, hình ảnh riêng tư hoặc thông tin cá nhân; không nhận quà để đổi lấy việc giữ bí mật.',
            'Nếu an toàn, giữ lại tin nhắn và tài khoản làm bằng chứng; nhờ người lớn hỗ trợ chặn hoặc báo cáo tài khoản.',
            'Nếu người đó tìm đến, theo dõi hoặc đe dọa, đến nơi có người lớn hỗ trợ ngay; gọi 113 khi nguy hiểm trực tiếp, hoặc 111 để được tư vấn bảo vệ trẻ em.'
        ])
    if online_person and meet_request:
        return ('CẦN XÁC MINH', 'Lời mời gặp mặt từ người chỉ quen trên mạng cần được kiểm tra với người lớn trước; không thể xác minh người đó qua lời kể.', [
            'Không tự đi gặp người chỉ quen trên mạng hoặc cung cấp địa chỉ, lịch trình cá nhân.',
            'Nói với cha mẹ hoặc thầy cô và nhờ họ kiểm tra, hỗ trợ quyết định.',
            'Nếu người đó yêu cầu giấu gia đình, dụ bằng quà hoặc gây áp lực, ngừng liên hệ và báo ngay cho người lớn.'
        ])
    if meet_request and secrecy:
        return ('NGUY CƠ CAO', 'Lời mời đi gặp kèm yêu cầu giấu người lớn là dấu hiệu đáng lo ngại; cần ưu tiên an toàn.', [
            'Không đi gặp riêng hoặc giữ bí mật cuộc hẹn với người lớn.',
            'Báo ngay cho cha mẹ hoặc giáo viên đáng tin cậy và cho họ xem lời nhắn.',
            'Không chia sẻ địa chỉ, vị trí hoặc lịch trình; tìm nơi an toàn nếu bị ép buộc.'
        ])

    # Phân loại nguy cơ cao trước bắt nạt thông thường; hỗ trợ nhiều cách diễn đạt.
    violence_phrases = (
        'de doa danh', 'doa danh', 'se danh', 'danh em', 'danh toi', 'danh no',
        'doi danh', 'chan danh', 'bi danh', 'danh hoi dong', 'cam dao',
        'dao de doa', 'bat coc', 'giet em', 'giet toi', 'danh sau gio hoc',
        'ep di theo', 'cho danh', 'hen danh', 'xu em', 'xu toi',
        'danh chet', 'dap em', 'danh vo mat', 'doi o cong truong',
        'chan duong danh', 'cam gay danh', 'doa giet', 'se giet',
        'danh cho mot tran', 'tim em de danh', 'danh sau tan hoc',
        'khong duoc mach', 'neu mach co', 'neu noi voi co', 'danh neu'
    )
    if contains(*violence_phrases) or (
        contains('de doa', 'doa nat', 'uy hiep') and
        contains('danh', 'bao luc', 'giet', 'chan duong', 'sau gio hoc')
    ):
        online_harassment = contains('facebook', 'dang anh', 'lay anh', 'che gieu', 'mang xa hoi', 'tiktok', 'phat tan', 'dang bai')
        explanation = ('Có dấu hiệu đe dọa bạo lực' +
            (' kết hợp bắt nạt hoặc đăng hình ảnh trên mạng' if online_harassment else '') +
            '. An toàn của em cần được ưu tiên; ứng dụng không thể xác minh mức độ nguy hiểm thực tế.')
        steps = [
            'Không gặp riêng hoặc tự đối chất với người đe dọa; tránh đi một mình, nhất là sau giờ học.',
            'Báo ngay cho cha mẹ, giáo viên chủ nhiệm hoặc người lớn đáng tin cậy; nhờ người lớn giúp em có nơi an toàn và phương án đi lại.',
            'Nếu an toàn, lưu tin nhắn, ảnh chụp màn hình, thời gian và đường dẫn bài đăng làm bằng chứng; không chia sẻ lại hình ảnh gây tổn thương.',
        ]
        if online_harassment:
            steps.append('Nhờ người lớn hỗ trợ báo cáo bài đăng hoặc tài khoản vi phạm trên nền tảng; không bình luận đối đầu hay lan truyền bài đăng.')
        steps.append('Nếu đang có nguy hiểm trực tiếp, đến nơi có người hỗ trợ và gọi 113; trẻ em có thể gọi 111 để được tư vấn, bảo vệ.')
        return ('NGUY CƠ CAO', explanation, steps)
    if contains('tong tien', 'doa tung anh', 'phat tan anh', 'anh nhay cam', 'anh rieng tu', 'quay clip', 'bi ep gui anh', 'doa dang anh', 'doa dua anh len mang', 'ep gui hinh', 'ep gui clip'):
        return ('NGUY CƠ CAO', 'Có dấu hiệu đe dọa, ép buộc hoặc xâm phạm quyền riêng tư.', [
            'Không gửi thêm ảnh, tiền hoặc thông tin riêng tư; không tự thương lượng một mình.',
            'Lưu bằng chứng ở nơi an toàn, chặn và báo cáo tài khoản sau khi đã lưu bằng chứng.',
            'Báo ngay cho cha mẹ, giáo viên hoặc người lớn đáng tin cậy; có thể gọi 111 để được hướng dẫn.'
        ])
    if contains('ep gui anh nhay cam', 'ep gui anh khoa than', 'doi anh nong', 'doa tung clip', 'doa tung hinh', 'ep quay clip', 'ep chup anh nhay cam'):
        return ('NGUY CƠ CAO', 'Có dấu hiệu ép buộc hoặc đe dọa liên quan đến hình ảnh riêng tư.', [
            'Không gửi thêm ảnh, clip, tiền hoặc thông tin cá nhân; không gặp riêng hay tự thương lượng với người đe dọa.',
            'Lưu bằng chứng tin nhắn và tài khoản ở nơi an toàn, nhưng không phát tán hình ảnh nhạy cảm.',
            'Báo ngay cho người lớn đáng tin cậy; trẻ em có thể gọi 111. Nếu nguy hiểm trực tiếp, gọi 113.'
        ])
    if contains('bat nat', 'co lap', 'tay chay', 'che gieu', 'noi xau', 'lang ma', 'xuc pham', 'chui boi', 'bi treu', 'bi choc'):
        return ('CẦN QUAN TÂM', 'Có dấu hiệu bắt nạt hoặc gây tổn thương trong quan hệ bạn bè.', [
            'Em không cần chịu đựng một mình. Hãy chia sẻ với giáo viên chủ nhiệm, cha mẹ hoặc người lớn đáng tin cậy.',
            'Ghi lại sự việc, lưu tin nhắn hoặc đường dẫn bài đăng nếu có; nhờ người lớn hỗ trợ báo cáo nội dung vi phạm, tránh đáp trả bằng xúc phạm hoặc bạo lực.',
            'Nếu có đe dọa hoặc nguy cơ bị đánh, hãy ưu tiên đến nơi an toàn và báo ngay cho người lớn.'
        ])
    # Cuộc gọi lạ, im lặng, gọi liên tục: không tự kết luận là lừa đảo.
    if contains('goi dien', 'cuoc goi', 'so la goi', 'so dien thoai la', 'dien thoai la', 'nhay may', 'goi nhieu lan', 'goi lien tuc', 'goi hoai', 'goi ma khong noi', 'khong nghe ai noi', 'dau so', 'so may la'):
        return ('CẦN QUAN TÂM', 'Em mô tả những cuộc gọi bất thường hoặc làm phiền. Chưa đủ căn cứ kết luận người gọi là ai hay có lừa đảo hay không.', [
            'Không gọi lại số lạ chỉ vì tò mò; không cung cấp mã OTP, mật khẩu, địa chỉ hoặc thông tin cá nhân.',
            'Nếu tiếp tục bị làm phiền, ghi lại thời điểm và tần suất cuộc gọi, chặn số và báo cuộc gọi rác qua chức năng của điện thoại hoặc nhà mạng.',
            'Nhờ cha mẹ hoặc người lớn đáng tin cậy hỗ trợ. Nếu người gọi đe dọa, tống tiền hoặc biết thông tin riêng tư của em, hãy báo ngay cho người lớn và giữ bằng chứng.'
        ])
    if contains('gia danh', 'mao danh', 'cong an goi', 'nhan vien ngan hang', 'nhan vien giao hang', 'yeu cau cai app', 'cai ung dung la', 'quet ma qr', 'qr la'):
        return ('CẦN XÁC MINH', 'Có dấu hiệu liên hệ giả danh hoặc yêu cầu thao tác có thể gây rủi ro.', [
            'Không làm theo yêu cầu cài ứng dụng, quét mã QR, chuyển tiền hoặc cung cấp OTP qua cuộc gọi hay tin nhắn chưa xác minh.',
            'Ngắt liên lạc và tự tìm số điện thoại chính thức của cơ quan, đơn vị để xác minh; không gọi theo số người lạ cung cấp.',
            'Nhờ cha mẹ hoặc thầy cô kiểm tra cùng. Nếu đã cung cấp thông tin tài khoản, báo ngay cho ngân hàng hoặc đơn vị liên quan qua kênh chính thức.'
        ])
    if contains('bi theo doi', 'theo doi em', 'nguoi la bam theo', 'nguoi la ru di', 'ru len xe', 'hen gap mot minh'):
        return ('NGUY CƠ CAO', 'Có dấu hiệu tiếp cận hoặc theo dõi khiến em có thể không an toàn.', [
            'Không đi theo người lạ, không đến điểm hẹn một mình; di chuyển đến nơi đông người hoặc nơi có người lớn đáng tin cậy.',
            'Báo ngay cho cha mẹ, thầy cô hoặc bảo vệ gần nhất; chia sẻ vị trí với người thân qua kênh riêng.',
            'Nếu có nguy hiểm trực tiếp, nhờ người lớn gọi 113; trẻ em có thể gọi 111 để được hỗ trợ.'
        ])
    if contains('bi hack', 'mat nick', 'chiem tai khoan', 'lo mat khau', 'lo otp', 'tai khoan bi danh cap'):
        return ('CẦN XÁC MINH', 'Có dấu hiệu tài khoản hoặc thông tin đăng nhập bị xâm phạm.', [
            'Đổi mật khẩu ngay từ thiết bị tin cậy, đăng xuất các phiên lạ và bật xác thực hai bước nếu còn truy cập được.',
            'Không cung cấp thêm mã OTP hay mật khẩu cho người tự xưng hỗ trợ; liên hệ bộ phận hỗ trợ chính thức của dịch vụ.',
            'Báo cho cha mẹ hoặc thầy cô nếu có tin nhắn mạo danh gửi từ tài khoản của em.'
        ])
    if contains('otp', 'mat khau', 'chuyen tien', 'trung thuong', 'hoc bong', 'link la', 'duong link', 'lua dao', 'tai khoan bi khoa', 'ma xac thuc'):
        return ('CẦN XÁC MINH', 'Nội dung có thể liên quan đến nguy cơ lừa đảo hoặc lộ thông tin trên mạng.', [
            'Không cung cấp mã OTP, mật khẩu, giấy tờ cá nhân hoặc chuyển tiền khi chưa xác minh.',
            'Không bấm đường dẫn lạ. Hãy kiểm tra bằng kênh chính thức và hỏi thầy cô hoặc cha mẹ.',
            'Em có thể dùng mục “Kiểm tra tin” để xem thêm các dấu hiệu lừa đảo; kết quả chỉ để tham khảo.'
        ])
    if contains('chuyen tiep anh', 'phat tan anh', 'ghep mat', 'ghep anh') and contains('ban', 'nguoi', 'hoc sinh', 'nhay cam'):
        return ('CẦN QUAN TÂM', 'Có thể liên quan đến phát tán hình ảnh hoặc xâm phạm quyền riêng tư.', [
            'Không chia sẻ tiếp ảnh hoặc nội dung gây tổn thương.',
            'Nhờ người lớn đáng tin cậy hỗ trợ báo cáo nội dung và bảo vệ người bị ảnh hưởng.'])
    if contains('ung dung la', 'file apk', 'tap tin la', 'phan mem la') and contains('cai', 'tai', 'cho phep', 'xin quyen'):
        return ('CẦN XÁC MINH', 'Ứng dụng hoặc tệp không rõ nguồn gốc có thể gây rủi ro cho thiết bị và dữ liệu.', [
            'Không cài đặt hoặc cấp quyền trước khi xác minh nguồn.',
            'Hỏi thầy cô hoặc người lớn và chỉ dùng ứng dụng từ nguồn chính thức.'])
    if contains('tin gia', 'video ai', 'gia giong', 'deepfake', 'tin don', 'thong bao nghi hoc'):
        return ('CẦN XÁC MINH', 'Thông tin có thể chưa được xác minh hoặc bị chỉnh sửa, giả mạo.', [
            'Không chia sẻ vội; kiểm tra nguồn và đối chiếu với thông báo chính thức.',
            'Hỏi giáo viên hoặc người lớn nếu thông tin liên quan đến nhà trường hoặc an toàn.'])
    if contains('buon', 'lo lang', 'so hai', 'ap luc', 'stress', 'khong ai hieu', 'co don', 'mat ngu'):
        return ('CẦN QUAN TÂM', 'Em có thể đang gặp chuyện khiến mình buồn hoặc lo lắng.', [
            'Hãy chia sẻ cụ thể với cha mẹ, thầy cô, nhân viên tư vấn học đường hoặc người lớn em tin tưởng.',
            'Nếu em thấy không an toàn hoặc có ý nghĩ làm hại bản thân, hãy tìm người lớn hỗ trợ ngay.',
            'Ứng dụng không thể đánh giá tình trạng tâm lý hay thay thế người tư vấn chuyên môn.'
        ])
    # Nhóm tích cực: chỉ xét sau khi đã loại trừ các dấu hiệu nguy cơ ở trên.
    # Không coi một lời mời học nhóm thông thường là dấu hiệu đe dọa.
    study_invite = contains('hoc nhom', 'cung hoc', 'on bai chung', 'ru em hoc', 'ru toi hoc', 'lam bai tap chung')
    friendly_context = contains('ban', 'ban hoc', 'ban cung lop', 'ban than', 'co giao', 'thay giao', 'chuc mung', 'duoc khen')
    unknown_contact = contains('nguoi la', 'nguoi moi quen tren mang', 'khong quen', 'chua tung gap', 'nguoi quen qua mang', 'tai khoan la')
    suspicious_invite = contains('mot minh', 'noi vang', 'khach san', 'nha nghi', 'giau bo me', 'khong cho ai biet', 'sau nua dem', 'chuyen tien', 'gui anh')
    if study_invite and friendly_context and not unknown_contact and not suspicious_invite:
        return ('TÌNH HUỐNG BÌNH THƯỜNG', 'Đây có vẻ là lời mời học nhóm tích cực, chưa thấy dấu hiệu nguy hiểm trong thông tin em cung cấp.', [
            'Em có thể đồng ý nếu biết rõ người bạn và cảm thấy thoải mái; em cũng có quyền từ chối nếu không muốn.',
            'Thống nhất thời gian và địa điểm học phù hợp, ưu tiên trường học, thư viện hoặc nơi quen thuộc; báo cho gia đình khi cần.',
            'Nếu lời mời thay đổi bất thường, có yêu cầu giữ bí mật hoặc khiến em không an tâm, hãy hỏi người lớn đáng tin cậy trước.'
        ])
    if contains('duoc khen', 'diem tot', 'diem cao', 'chuc mung', 'giup do ban', 'lam viec tot', 'tham gia cau lac bo') and not unknown_contact and not suspicious_invite:
        return ('TÌNH HUỐNG BÌNH THƯỜNG', 'Em đang kể về một trải nghiệm học tập hoặc quan hệ bạn bè tích cực; chưa thấy dấu hiệu nguy hiểm trong mô tả.', [
            'Em có thể vui vẻ đón nhận lời khen và tiếp tục phát huy những việc làm tích cực.',
            'Trao đổi lịch sự, tôn trọng sự tự nguyện của mình và của bạn bè.',
            'Nếu xuất hiện điều khiến em khó chịu hoặc không an toàn, hãy chia sẻ với người lớn đáng tin cậy.'
        ])
    if study_invite and unknown_contact:
        return ('CẦN XÁC MINH', 'Lời mời học nhóm đến từ người chưa quen rõ ngoài đời; cần xác minh trước khi gặp mặt.', [
            'Không hẹn gặp riêng người mới quen qua mạng và không chia sẻ địa chỉ, lịch trình hay thông tin cá nhân.',
            'Hỏi cha mẹ hoặc thầy cô; nếu thực sự cần học nhóm, ưu tiên nhóm có bạn học quen biết và địa điểm công khai.',
            'Nếu người đó thúc ép giữ bí mật, gửi ảnh hoặc chuyển tiền, hãy dừng liên hệ và báo người lớn.'
        ])
    # Đối chiếu 100 ví dụ minh họa khi người dùng nhập đúng nội dung mẫu.
    # Không dùng phép so khớp gần đúng để suy diễn an toàn với tình huống mới.
    for sample, category, explanation, safe_action in DETECTIVE:
        if normalize(message).strip() == normalize(sample).strip():
            level = ('NGUY CƠ CAO' if category == 'Không đáng tin cậy' else
                     'CẦN XÁC MINH' if category == 'Cần xác minh thêm' else
                     'TÌNH HUỐNG BÌNH THƯỜNG')
            return (level, explanation, [safe_action, 'Nếu còn băn khoăn hoặc cảm thấy không an toàn, hãy hỏi người lớn đáng tin cậy.'])
    return ('CHƯA ĐỦ THÔNG TIN', 'Ứng dụng chưa nhận diện chắc chắn tình huống của em; điều này không có nghĩa là an toàn.', [
        'Hãy nói rõ sự việc với thầy cô, cha mẹ hoặc người lớn đáng tin cậy để được hỗ trợ phù hợp.',
        'Không đưa thông tin cá nhân thật lên ứng dụng. Nếu có đe dọa hoặc nguy hiểm, hãy tìm sự trợ giúp ngay.',
        'Ứng dụng hiện chỉ tư vấn theo các nhóm dấu hiệu đã lập trình, chưa hiểu được mọi cách diễn đạt.'
    ])

SHARING_EXAMPLES_100 = [item[0] for item in DETECTIVE]

with tab6:
    st.subheader('💬 Góc chia sẻ – Tình huống của em')
    st.write('Em có thể tự viết tình huống đang băn khoăn để nhận hướng dẫn an toàn ban đầu.')
    st.warning('Không nhập họ tên thật, số điện thoại, địa chỉ, tên trường/lớp, mật khẩu, OTP hoặc thông tin nhận dạng người khác. Không gửi ảnh riêng tư. Nội dung có thể được xử lý trên máy chủ của ứng dụng.')
    with st.form('student_free_situation_form'):
        student_story = st.text_area('Em muốn chia sẻ điều gì?', height=155, max_chars=1500,
            placeholder='Ví dụ: Có bạn nhắn sẽ chặn đánh em sau giờ học nếu em nói với cô giáo...')
        ask = st.form_submit_button('🛡️ NHẬN HƯỚNG DẪN', type='primary')
    if ask:
        if len(student_story.strip()) < 12:
            st.warning('Em hãy mô tả rõ hơn một chút (ít nhất 12 ký tự), không cần nêu tên thật nhé.')
        else:
            level, explanation, steps = advise_student_situation(student_story)
            if level in ('CẦN HỖ TRỢ NGAY', 'NGUY CƠ CAO'):
                st.error('🚨 ' + level)
            elif level in ('CHƯA ĐỦ THÔNG TIN', 'TÌNH HUỐNG BÌNH THƯỜNG'):
                st.info('ℹ️ ' + level)
            else:
                st.warning('⚠️ ' + level)
            st.write('**Nhận định ban đầu:** ' + explanation)
            st.markdown('#### Em nên làm gì?')
            for index, step in enumerate(steps, 1):
                st.write(f'{index}. {step}')
            st.caption('Đây là hướng dẫn theo quy tắc, không phải tư vấn trực tiếp của chuyên gia và không bảo đảm phát hiện mọi nguy cơ. 111: Tổng đài bảo vệ trẻ em; 113: Công an; 115: Cấp cứu y tế (Việt Nam).')

with tab5:
    st.subheader('ℹ️ Giới thiệu phần mềm')
    st.caption('Phiên bản 4.8 – Đồng bộ cảnh báo và cải thiện giao diện')
    st.write('**Mục tiêu:** Giúp học sinh THCS nhận diện dấu hiệu lừa đảo và luyện kỹ năng ứng xử an toàn trên mạng.')
    st.markdown('**Các hoạt động chính**')
    st.write('🔎 **Kiểm tra tin:** Nhận cảnh báo và hướng dẫn xác minh tin nhắn.')
    st.write('🎮 **Thử thách:** Mỗi lượt trả lời 10 câu hỏi từ ngân hàng 34 câu.')
    st.write('🕵️ **Thám tử & Tình huống:** Ngân hàng 100 tin nhắn, mỗi lượt chơi 10 câu ngẫu nhiên; luyện xử lý 20 tình huống.')
    st.write('💬 **Góc chia sẻ:** Học sinh tự nhập tình huống và nhận hướng dẫn ban đầu theo quy tắc; 100 ví dụ tham khảo không có nghĩa phần mềm hiểu chính xác mọi câu tự do.')
    st.markdown('**Công nghệ và giới hạn**')
    st.write('Phần mềm sử dụng Python, Streamlit và mô hình TF-IDF kết hợp Logistic Regression cùng các quy tắc nhận diện.')
    st.warning('Kết quả chỉ để tham khảo. Dữ liệu huấn luyện còn ít và chủ yếu là tình huống giả lập; phần mềm có thể cảnh báo nhầm hoặc bỏ sót nguy cơ. Không dùng kết quả để kết luận chắc chắn một tin nhắn an toàn.')
    st.write('**Kiểm thử:** Bộ số 4 đã được dùng trong quá trình điều chỉnh nên không phải kiểm thử độc lập. Bộ số 5 gồm các tình huống giả lập mới; kết quả chưa phản ánh độ chính xác ngoài thực tế. Tài liệu kiểm thử được lưu riêng để phục vụ nghiên cứu.')
    st.write('**Quyền riêng tư:** Không nhập mật khẩu, mã xác thực OTP hay thông tin cá nhân thật. Bản chạy cục bộ không chủ động gửi tin nhắn đến dịch vụ AI bên ngoài.')
    st.caption('Phần mềm hỗ trợ giáo dục và nghiên cứu; không thay thế việc xác minh thực tế hoặc sự hỗ trợ của thầy cô, cha mẹ.')
