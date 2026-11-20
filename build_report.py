# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os
from pathlib import Path

BASE = str(Path(__file__).resolve().parent)
REPORT_DIR = os.path.join(BASE, "01_Report")
FIGS = os.path.join(BASE, "03_SourceCode", "notebooks", "figures")
OUT = os.path.join(REPORT_DIR, "BaoCao_Nhom_Framingham_TV2.docx")

doc = Document()

# Base style
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(13)
style.paragraph_format.line_spacing = 1.3
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.first_line_indent = Inches(0.3)

def set_cell_bg(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
    return h

def para(text, bold=False, italic=False, align=None, size=None, space_after=6):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return p

def eq(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = 'Cambria Math'
    run.font.size = Pt(12)
    run.italic = False
    p.paragraph_format.space_after = Pt(4)
    return p

def bullet(text):
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    return p

def add_image(path, caption, width=6.0):
    if not os.path.exists(path):
        para(f"[Thiếu ảnh: {path}]", italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    r.add_picture(path, width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(11)
    cap.paragraph_format.space_after = Pt(10)

def add_table(headers, rows, caption, widths=None):
    cap = doc.add_paragraph()
    run = cap.add_run(caption)
    run.bold = True
    run.font.size = Pt(11)
    cap.paragraph_format.space_after = Pt(4)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ''
        r = hdr[i].paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(10)
        set_cell_bg(hdr[i], 'D9E2F3')
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(10)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

# ---------------- Title ----------------
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('BÁO CÁO DỰ ÁN: DỰ ĐOÁN BỆNH TIM (FRAMINGHAM HEART DISEASE PREDICTION)')
r.bold = True
r.font.size = Pt(15)
p.paragraph_format.space_after = Pt(2)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('Phần thực hiện của Thành viên 2 (TV2): Nền tảng lý thuyết & Kết quả thực nghiệm')
r.italic = True
r.font.size = Pt(12)
p.paragraph_format.space_after = Pt(12)

# ---------------- Chapter 3 ----------------
add_heading('Chương 3. Cơ sở lý thuyết', level=1)

add_heading('3.1. Logistic Regression', level=2)
para('Logistic Regression là mô hình phân loại nhị phân phổ biến, ước lượng xác suất mẫu thuộc lớp dương (y=1) dựa trên tổ hợp tuyến tính của các đặc trưng đầu vào. Mô hình không dự đoán nhãn trực tiếp mà dự đoán xác suất, sau đó so sánh với ngưỡng để đưa ra quyết định phân loại.')
para('Bộ dự đoán tuyến tính (linear predictor):')
eq('z = β0 + Σ_{j=1}^{p} βj xj')
para('Trong đó z là giá trị trước kích hoạt, β0 là hệ số chặn (bias), βj là trọng số của đặc trưng thứ j, xj là giá trị đặc trưng thứ j, p là số lượng đặc trưng.')
para('Hàm sigmoid ánh xạ giá trị thực z vào khoảng (0,1):')
eq('σ(z) = 1 / (1 + e^{-z})')
para('Xác suất mẫu thuộc lớp dương:')
eq('P(y=1|x) = σ(z)')
para('Quyết định phân loại: nếu P(y=1|x) ≥ 0.5 thì dự đoán y=1, ngược lại y=0.')
para('Hàm mất mát Binary Cross-Entropy (Log-Loss):')
eq('L = - (1/N) Σ_{i=1}^{N} [ yi log(ŷi) + (1-yi) log(1-ŷi) ]')
para('Trong đó N là số mẫu, yi là nhãn thực, ŷi là xác suất dự đoán.')
para('Chuẩn hóa L1 (Lasso) và L2 (Ridge) được thêm vào hàm mất mát để kiểm soát độ phức tạp mô hình và tránh overfitting. L1 có thể đưa một số trọng số về 0 (chọn lọc đặc trưng), L2 thu nhỏ trọng số đồng đều.')
para('Tập dữ liệu Framingham mất cân bằng lớp (tỷ lệ dương ~15%). Mô hình sử dụng class_weight=\'balanced\' để điều chỉnh trọng số lớp, giúp mô hình nhạy hơn với lớp thiểu số (nguy cơ bệnh tim), cải thiện Recall.')
para('Ưu điểm: đơn giản, diễn giải tốt, huấn luyện nhanh, cung cấp xác suất. Hạn chế: giả định quan hệ tuyến tính giữa log-odds và đặc trưng; nhạy với đa cộng tuyến; có thể underfit nếu dữ liệu phức tạp.')

add_heading('3.2. K-Nearest Neighbors (k-NN)', level=2)
para('k-NN là thuật toán phân loại dựa trên khoảng cách. Với một mẫu mới, thuật toán tìm k mẫu huấn luyện gần nhất và bỏ phiếu theo đa số để quyết định nhãn.')
para('Khoảng cách Euclidean:')
eq('d_Euclidean(x,y) = sqrt( Σ_{j=1}^{p} (xj - yj)^2 )')
para('Khoảng cách Manhattan:')
eq('d_Manhattan(x,y) = Σ_{j=1}^{p} |xj - yj|')
para('Khoảng cách Minkowski (tổng quát):')
eq('d_Minkowski(x,y) = ( Σ_{j=1}^{p} |xj - yj|^q )^{1/q}')
para('Trong đó q=1 tương ứng Manhattan, q=2 tương ứng Euclidean.')
para('Bỏ phiếu đồng đều (uniform): mỗi láng giềng có trọng số bằng nhau. Bỏ phiếu theo khoảng cách (distance): láng giềng gần hơn có trọng số lớn hơn, thường là nghịch đảo khoảng cách.')
para('Ảnh hưởng của k: k nhỏ → mô hình phức tạp, dễ overfit; k lớn → mô hình mượt hơn, dễ underfit. Cần chuẩn hóa đặc trưng (scaling) vì k-NN dựa trên khoảng cách; nếu không chuẩn hóa, đặc trưng có thang lớn sẽ chi phối.')
para('Trong thực nghiệm, k-NN sử dụng metric Minkowski với p=1 (Manhattan) và p=2 (Euclidean), weights uniform/distance, n_neighbors ∈ {3,5,7,9,11,15,21}. Kết quả tốt nhất: n_neighbors=21, weights=distance, p=2 (Euclidean), CV ROC-AUC = 0.6913.')
para('Ưu điểm: đơn giản, không giả định phân phối, thích nghi tốt dữ liệu phi tuyến. Hạn chế: chậm khi dự đoán, nhạy với chiều cao, nhạy với nhiễu và mất cân bằng lớp.')

add_heading('3.3. Support Vector Machine (SVM)', level=2)
para('SVM tìm siêu phẳng (hyperplane) phân tách hai lớp với biên (margin) lớn nhất. Các điểm gần biên nhất (support vectors) quyết định siêu phẳng.')
para('Hàm quyết định:')
eq('f(x) = w^T x + b')
para('Bài toán tối ưu soft-margin:')
eq('min_{w,b,ξ} (1/2)||w||^2 + C Σ_{i=1}^{N} ξi')
para('Ràng buộc:')
eq('yi (w^T xi + b) ≥ 1 - ξi,   ξi ≥ 0')
para('Trong đó w là vector trọng số, b là bias, ξi là biến lỏng (slack) cho phép vi phạm margin, C là tham số điều chuẩn.')
para('Kernel tuyến tính:')
eq('K(xi, xj) = xi^T xj')
para('Kernel RBF (Radial Basis Function):')
eq('K(xi, xj) = exp( -γ ||xi - xj||^2 )')
para('Tham số C kiểm soát đánh đổi giữa margin lớn và lỗi phân loại: C lớn → ít vi phạm, dễ overfit; C nhỏ → margin lớn hơn, tổng quát hóa tốt hơn. Gamma trong RBF kiểm soát phạm vi ảnh hưởng của mỗi mẫu: gamma lớn → biên quyết định phức tạp, dễ overfit; gamma nhỏ → mượt hơn.')
para('Trong thực nghiệm: Linear SVM C ∈ {0.01,0.1,1,10,100}, tốt nhất C=0.1 (CV ROC-AUC=0.7285). RBF SVM C ∈ {0.1,1,10,100}, gamma ∈ {scale,0.001,0.01,0.1,1}, tốt nhất C=0.1, gamma=0.1 (CV ROC-AUC=0.6496).')
para('Ưu điểm: hiệu quả trong không gian nhiều chiều, tối đa margin, kernel trick xử lý phi tuyến. Hạn chế: chậm với dữ liệu lớn, nhạy với lựa chọn C và gamma, khó diễn giải.')

# ---------------- Chapter 4.1 ----------------
add_heading('Chương 4. Kết quả thực nghiệm và đánh giá', level=1)
add_heading('4.1. Kết quả thực nghiệm và đánh giá các mô hình Logistic Regression, k-NN và SVM', level=2)

add_heading('4.1.1. Phương pháp thực nghiệm', level=3)
para('Dữ liệu sử dụng là Framingham Heart Study (framingham.csv). Biến mục tiêu là TenYearCHD (nguy cơ bệnh tim trong 10 năm). Dữ liệu được chia train/test theo tỷ lệ 80/20 với random_state=42 và stratify theo biến mục tiêu.')
para('Tiền xử lý theo TV1: xử lý giá trị thiếu, mã hóa biến phân loại, chuẩn hóa (scaling) các đặc trưng số, đóng gói trong Scikit-learn Pipeline để tránh rò rỉ dữ liệu (data leakage).')
para('Đánh giá: Stratified K-Fold 5 lớp (StratifiedKFold, n_splits=5, shuffle=True, random_state=42), tối ưu ROC-AUC. GridSearchCV cho k-NN và SVM. Môi trường: Python, Scikit-learn, Pandas, NumPy, Matplotlib/Seaborn.')

# Baseline table
add_table(
    ['Mô hình', 'CV ROC-AUC (Mean±Std)', 'Test ROC-AUC', 'Test Recall', 'Test F1', 'Test Precision', 'Test Accuracy'],
    [
        ['Logistic Regression', '0.7307±0.0302', '0.7002', '0.5969', '0.3557', '0.2533', '0.6710'],
    ],
    'Bảng 4.1. Kết quả Baseline Logistic Regression (class_weight=\'balanced\', max_iter=1000).'
)

add_heading('4.1.2. Logistic Regression Baseline', level=3)
para('Cấu hình: LogisticRegression(class_weight=\'balanced\', random_state=42, max_iter=1000) trong Pipeline với preprocessor. CV ROC-AUC = 0.7307±0.0302. Trên tập test: ROC-AUC=0.7002, Recall=0.5969, F1=0.3557, Precision=0.2533, Accuracy=0.6710. Mô hình có Recall cao nhất trong các mô hình, phù hợp bài toán sàng lọc nguy cơ.')
add_image(os.path.join(FIGS, 'baseline_roc_curve.png'), 'Hình 4.1. ROC Curve - Logistic Regression Baseline (Test set).', width=5.0)
add_image(os.path.join(FIGS, 'baseline_confusion_matrix.png'), 'Hình 4.2. Confusion Matrix - Logistic Regression Baseline.', width=5.0)

add_heading('4.1.3. Tối ưu k-NN', level=3)
para('Không gian tìm kiếm: n_neighbors ∈ {3,5,7,9,11,15,21}, weights ∈ {uniform, distance}, metric=minkowski, p ∈ {1,2}. Tổng 28 tổ hợp, GridSearchCV 5-fold.')
para('Kết quả tốt nhất: n_neighbors=21, weights=distance, p=2 (Euclidean), CV ROC-AUC=0.6913±0.0385. Trên tập test: ROC-AUC=0.6303, Recall=0.0310, F1=0.0567, Precision=0.3333, Accuracy=0.8432.')
para('k lớn (21) và distance weighting giúp mô hình mượt, nhưng Recall rất thấp do mất cân bằng lớp; k-NN thiên về lớp đa số.')

add_heading('4.1.4. Tối ưu Linear SVM', level=3)
para('Kernel tuyến tính, C ∈ {0.01,0.1,1,10,100}. Kết quả tốt nhất C=0.1, CV ROC-AUC=0.7285±0.0298. Test: ROC-AUC=0.6965, Recall=0.0698, F1=0.1233, Precision=0.5294, Accuracy=0.8491.')

add_heading('4.1.5. Tối ưu RBF SVM', level=3)
para('Kernel RBF, C ∈ {0.1,1,10,100}, gamma ∈ {scale,0.001,0.01,0.1,1}. Kết quả tốt nhất C=0.1, gamma=0.1, CV ROC-AUC=0.6496±0.0343. Test: ROC-AUC=0.5378, Recall=0.0, F1=0.0, Precision=0.0, Accuracy=0.8479. Mô hình gần như không phát hiện lớp dương (underfitting/collapse).')

add_heading('4.1.6. So sánh mô hình', level=3)
rows = [
    ['Logistic Regression', '0.7307±0.0302', '0.7002', '0.6816±0.0390', '0.5969', '0.3893±0.0320', '0.3557', '0.2728±0.0264', '0.2533', '0.6740±0.0288', '0.6710'],
    ['k-NN', '0.6913±0.0385', '0.6303', '0.0388±0.0194', '0.0310', '0.0713±0.0335', '0.0567', '0.5371±0.1288', '0.3333', '0.8491±0.0022', '0.8432'],
    ['Linear SVM', '0.7285±0.0298', '0.6965', '0.0835±0.0323', '0.0698', '0.1436±0.0440', '0.1233', '0.6311±0.1304', '0.5294', '0.8520±0.0039', '0.8491'],
    ['RBF SVM', '0.6496±0.0343', '0.5378', '0.0000±0.0000', '0.0000', '0.0000±0.0000', '0.0000', '0.0000±0.0000', '0.0000', '0.8482±0.0001', '0.8479'],
]
add_table(
    ['Mô hình', 'CV ROC-AUC', 'Test ROC-AUC', 'CV Recall', 'Test Recall', 'CV F1', 'Test F1', 'CV Precision', 'Test Precision', 'CV Acc', 'Test Acc'],
    rows,
    'Bảng 4.2. So sánh đầy đủ 4 mô hình (5-fold CV vs Holdout Test).',
    widths=[1.0,0.85,0.7,0.7,0.6,0.6,0.6,0.7,0.7,0.6,0.6]
)
para('Logistic Regression có CV ROC-AUC và Test ROC-AUC cao nhất, đồng thời Recall cao nhất. k-NN và SVM có Accuracy cao nhưng Recall rất thấp (thiên về lớp đa số), không phù hợp bài toán sàng lọc.')

add_heading('4.1.7. Phân tích Learning Curve', level=3)
for name, f in [('Logistic Regression','baseline_learning_curve.png'),('k-NN','knn_learning_curve.png'),('Linear SVM','svm_linear_learning_curve.png'),('RBF SVM','svm_rbf_learning_curve.png')]:
    add_image(os.path.join(FIGS, f), f'Hình. Learning Curve - {name}.', width=5.5)
para('Learning Curve cho thấy training score và CV score theo kích thước tập huấn luyện. Logistic Regression: khoảng cách nhỏ giữa train và CV → ít overfit, tổng quát hóa tốt. k-NN: train score cao (≈1.0 với distance) nhưng CV thấp → overfit. Linear SVM: train/CV gần nhau, ổn định. RBF SVM: CV score thấp và phẳng → underfitting, mô hình không học được mẫu lớp dương.')

add_heading('4.1.8. Phân tích Validation Curve', level=3)
add_image(os.path.join(FIGS, 'knn_validation_curve.png'), 'Hình. Validation Curve - k-NN: n_neighbors.', width=5.5)
add_image(os.path.join(FIGS, 'svm_linear_validation_curve.png'), 'Hình. Validation Curve - Linear SVM: C.', width=5.5)
add_image(os.path.join(FIGS, 'svm_rbf_validation_curve.png'), 'Hình. Validation Curve - RBF SVM: C (gamma tốt nhất).', width=5.5)
para('Validation Curve minh họa ảnh hưởng hyperparameter: k-NN CV tăng khi n_neighbors tăng đến ~21; Linear SVM CV cao nhất ở C=0.1; RBF SVM CV thấp với mọi C, cho thấy RBF không phù hợp dữ liệu này.')

add_heading('4.1.9. Thảo luận và kết luận', level=3)
para('Logistic Regression là mô hình tốt nhất theo ROC-AUC (CV 0.7307, Test 0.7002) và Recall (0.5969), phù hợp bài toán sàng lọc nguy cơ tim mạch. k-NN và SVM đạt Accuracy cao nhưng Recall thấp do mất cân bằng lớp. CV và holdout cho kết quả nhất quán. Hạn chế: mất cân bằng lớp, dữ liệu hạn chế, chỉ mang tính giáo dục, không dùng cho chẩn đoán lâm sàng.')

# References
add_heading('Tài liệu tham khảo', level=1)
refs = [
    'Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. Machine Learning, 20(3), 273-297.',
    'Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory, 13(1), 21-27.',
    'Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.',
    'Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression (3rd ed.). Wiley.',
]
for r in refs:
    para(r, size=12)

doc.save(OUT)
print('saved', OUT)
