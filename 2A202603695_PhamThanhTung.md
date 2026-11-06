# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

|                                 |   F1 test |   F1 SUB 40 |   p95 ms/tin |   USD/1k tin |
|:--------------------------------|----------:|------------:|-------------:|-------------:|
| 1. BoW + LogReg (tự code)       |    0.8817 |      0.8235 |       0.085  |            0 |
| 2. BoW + MLP (deep learning)    |    0.9314 |      0.9189 |       0.1862 |            0 |
| 3a. Encoder (MiniLM) + kNN      |    0.9009 |      0.9474 |      63.6968 |            0 |
| 3b. Encoder (MiniLM) + LogReg   |    0.9479 |      1      |      37.1839 |            0 |
| 3c. Encoder (MiniLM) + MLP head |    0.9569 |      1      |      37.6984 |            0 |
| 4a. LLM nhỏ zero-shot           |    0.9756 |      0.9756 |    2458.5    |            0 |
| 4b. LLM lớn zero-shot           |    1      |      1      |    2411.21   |            0 |
| 4c. LLM nhỏ few-shot            |    0.9524 |      0.9524 |    2499.1    |            0 |

## 3. Dự đoán và thực tế

ml_f1: Tôi dự đoán F1-score trên tập test khoảng 0.85. Precision có thể cao hơn Recall vì mô hình nhận diện khá chính xác những tin nhắn được dự đoán là spam, nhưng vẫn bỏ sót một số spam. F1 trên tập train có thể cao hơn tập test do mô hình đã học từ dữ liệu train.

ml_lr: Với lr=0.001, loss giảm rất chậm vì trọng số được cập nhật quá nhỏ. Với lr=0.5, loss dự kiến giảm đều và ổn định. Với lr=50, loss có thể dao động mạnh hoặc tăng đột ngột do bước cập nhật trọng số quá lớn, khiến mô hình vượt qua điểm tối ưu.

dl_curve: Tôi dự đoán train loss sẽ giảm dần sau 30 epoch. Validation loss ban đầu cũng giảm, nhưng có thể tăng trở lại khi mô hình bắt đầu overfit. F1-score của MLP có thể tương đương hoặc nhỉnh hơn Logistic Regression, nhưng chênh lệch không lớn vì dữ liệu nhỏ và đầu vào là Bag-of-Words.

dl_relu: Nếu bỏ ReLU, hai lớp Linear liên tiếp vẫn chỉ tạo ra một phép biến đổi tuyến tính (affine). Do đó mô hình tương đương về khả năng biểu diễn với Logistic Regression. Tôi dự đoán F1-score có thể tương đương Logistic Regression và không nhất thiết bằng MLP có ReLU, vì mô hình không còn khả năng học quan hệ phi tuyến.

emb_search: Tôi dự đoán phần lớn các tin nhắn gần nhất sẽ là Spam. Dù truy vấn không chứa các từ spam quen thuộc như free, win, prize, nó vẫn mang ý nghĩa thông báo trúng thưởng và yêu cầu phản hồi để nhận quà. Embedding có khả năng nhận diện sự tương đồng về ngữ nghĩa giữa các câu, ngay cả khi chúng sử dụng từ khác nhau.

emb_tricky: Tôi dự đoán Encoder + Logistic Regression sẽ phân loại đúng nhiều hơn Bag-of-Words. Vì Bag-of-Words dựa vào sự xuất hiện của từng từ nên có thể nhầm các tin Ham chứa từ free, won, prize thành Spam hoặc bỏ sót các tin Spam dùng từ thay thế. Embedding có khả năng biểu diễn ngữ nghĩa của toàn câu nên có thể phân biệt ngữ cảnh tốt hơn. Tuy nhiên, cả hai mô hình vẫn có thể dự đoán sai.

llm_models: Tôi dự đoán model lớn sẽ có F1 cao hơn model nhỏ khoảng 0.02 đến 0.05, nhưng có thể không chênh lệch nhiều vì phân loại Spam/Ham là nhiệm vụ tương đối đơn giản. Model lớn thường đắt hơn, tuy nhiên tỷ lệ chi phí phụ thuộc bảng giá và số token thực tế. LLM không chắc thắng Encoder + Logistic Regression (3b), vì mô hình 3b đã học trực tiếp trên dữ liệu Spam/Ham.

llm_fewshot: Tôi dự đoán Few-shot sẽ có F1 tương đương Zero-shot vì model nhỏ đã đạt F1=1.000 trên tập SUB. Few-shot không thể đạt F1 cao hơn 1.000. Tuy nhiên, kết quả cũng có thể giảm nếu các ví dụ không phù hợp. Chi phí có thể tăng vì prompt chứa thêm 4 ví dụ, làm tăng input token.

tradeoff: Tôi dự đoán BoW + Logistic Regression và một số mô hình Embedding có thể nằm trên đường Pareto vì có sự cân bằng giữa độ chính xác và tốc độ. Các mô hình LLM có khả năng nằm bên phải vạch SLA 50 ms do thời gian gọi API lớn. Tôi sẽ kiểm chứng dự đoán bằng biểu đồ thực tế.

hard_eval: Tôi dự đoán Accuracy của các mô hình sẽ giảm trên 50 case khó so với tập test thông thường. BoW có thể gặp khó khăn với Spam viết tự nhiên, các từ bị biến dạng và tin Ham chứa từ khóa Spam. Embedding có thể hiểu ngữ nghĩa tốt hơn nhưng vẫn nhầm ở các tình huống cần ngữ cảnh. LLM có thể gặp khó ở nhóm prompt injection. Tôi dự đoán các mô hình đạt khoảng 60-85% Accuracy, nhưng đây chỉ là giả thuyết cần kiểm chứng

## 4. Phân tích 50 case khó
1. Mô hình nào tụt F1 nhiều nhất? Ở nhóm case nào?

Trong quá trình kiểm thử, tôi nhận thấy nhóm natural_spam là một trong những nhóm có khả năng gây nhầm lẫn cho các mô hình phân loại Spam/Ham.

Các tin nhắn trong nhóm này thường được viết bằng giọng điệu tự nhiên, thân mật, giống như lời nhắn từ người quen hoặc thông báo hợp lệ. Chúng không nhất thiết chứa những từ khóa Spam phổ biến như "FREE", "WIN" hay "URGENT".

Đặc biệt, mô hình BoW + Logistic Regression có nguy cơ phân loại sai do chủ yếu dựa vào những đặc trưng từ vựng đã học. Trong khi đó, Embedding và LLM có thể khai thác ngữ nghĩa, nhưng vẫn có khả năng nhầm các tin nhắn lừa đảo được viết khéo léo.

Tuy nhiên, để kết luận mô hình nào tụt F1 nhiều nhất, cần đối chiếu F1 trên tập test thường với F1 trên 50 case khó của từng mô hình.

2. Phân tích hai case cụ thể

Case 1 — ID 12: Spam giả dạng tin nhắn làm quen

Nhóm: natural_spam

Nhãn thực tế: Spam

Nội dung: "hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club"

Phân tích:

Tin nhắn mở đầu bằng lời chào lịch sự, tự giới thiệu và đề cập đến một người bạn chung. Nội dung ban đầu khá giống một cuộc trò chuyện bình thường nên dễ bị nhận nhầm thành Ham.

Tuy nhiên, mục đích chính của tin nhắn là dẫn người nhận truy cập vào một trang web bên ngoài không được yêu cầu.

BoW có thể gặp khó khăn vì phần lớn từ ngữ là những từ giao tiếp thông thường. Embedding hoặc LLM có thể nhận ra mục đích dẫn dụ, nhưng vẫn có nguy cơ nhầm lẫn nếu không xem xét toàn bộ ngữ cảnh.

Case 2 — ID 21: Lừa đảo giả danh người thân

Nhóm: manipulation

Nhãn thực tế: Spam

Nội dung: "Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x"

Phân tích:

Đây là tin nhắn giả danh con của người nhận, thông báo vừa đổi số điện thoại và nhờ chuyển gấp 200 bảng Anh vào tài khoản của một người khác.

Tin nhắn không chứa các từ khóa Spam truyền thống mà sử dụng mối quan hệ gia đình để tạo lòng tin và thúc đẩy người nhận chuyển tiền.

BoW có thể gặp khó khăn vì các từ như "Mum", "phone" và "friend" thường xuất hiện trong những cuộc trò chuyện hợp lệ.

Embedding và LLM có khả năng nhận diện ngữ cảnh tốt hơn, nhưng việc phát hiện gian lận còn phụ thuộc vào khả năng nhận ra các dấu hiệu đáng ngờ: số điện thoại mới, yêu cầu chuyển tiền và tài khoản bên thứ ba.

3. Kết luận

Qua việc phân tích các case khó, tôi nhận thấy mô hình phân loại Spam không nên chỉ dựa vào từ khóa. Nhiều tin nhắn lừa đảo được viết giống những cuộc trò chuyện thông thường, khiến việc phân loại trở nên khó khăn hơn.

Các nhóm natural_spam và manipulation cho thấy tầm quan trọng của việc nhận diện ngữ cảnh, mục đích và hành vi lừa đảo.

Để đánh giá chính xác mô hình tốt nhất, cần kết hợp F1 trên tập test thông thường, kết quả stress test và phân tích các trường hợp dự đoán sai.

## 5. Thử thách cá nhân

### Đề #2: Machine Learning — Thay đổi kích thước Vocabulary

**Đề bài:** Giảm Vocabulary còn 100 từ và tăng lên 5.000 từ bằng `build_vocab(..., max_size)`. Huấn luyện lại mô hình Logistic Regression, báo cáo F1 trên tập test và giải thích xu hướng.

### 5.1. Cách thực hiện

Tôi thực hiện thí nghiệm bằng cách thay đổi kích thước Vocabulary để đánh giá ảnh hưởng đến hiệu quả phân loại Spam/Ham của mô hình **Bag-of-Words + Logistic Regression**.

Các bước thực hiện:

1. Sử dụng tập dữ liệu huấn luyện (`texts_tr`) và tập kiểm thử (`texts_te`) đã được chia sẵn.
2. Tạo hai bộ từ vựng bằng `build_vocab()` với `max_size=100` và `max_size=5000`.
3. Chuyển đổi dữ liệu văn bản thành vector BoW bằng hàm `vectorize()`.
4. Huấn luyện hai mô hình Logistic Regression với cùng tham số:
   - Learning rate: `0.5`
   - Epochs: `300`
   - Ngưỡng phân loại: `0.5`
5. Dự đoán trên tập test và tính F1-score bằng hàm `metrics()`.

### 5.2. Kết quả thực nghiệm

| Chỉ số | Vocabulary 100 từ | Vocabulary 5.000 từ |
|---|---:|---:|
| Số chiều đầu vào | 100 | 5.000 |
| Số lượng train | 3.619 | 3.619 |
| Số lượng test | 775 | 775 |
| Learning rate | 0.5 | 0.5 |
| Epochs | 300 | 300 |
| Loss cuối | 0.1281 | 0.0826 |
| **F1-score (test)** | **0.8222** | **0.8757** |

**Chênh lệch F1-score:**

\[
\Delta F1 = 0.8757 - 0.8222 = 0.0535
\]

Như vậy, khi tăng Vocabulary từ 100 lên 5.000 từ, F1-score tăng **0.0535**, tương đương **5.35 điểm phần trăm**.

### 5.3. Phân tích kết quả

**Trường hợp Vocabulary 100 từ:**

- Số lượng từ vựng nhỏ khiến mô hình chỉ sử dụng được một phần thông tin trong văn bản.
- Nhiều từ quan trọng để phân biệt Spam và Ham có thể bị loại bỏ.
- Mô hình có ít đặc trưng hơn nên khả năng phân loại bị hạn chế.
- Kết quả F1-score đạt `0.8222`.

**Trường hợp Vocabulary 5.000 từ:**

- Mô hình có nhiều đặc trưng hơn để biểu diễn nội dung tin nhắn.
- Khả năng giữ lại các từ khóa và mẫu từ đặc trưng của Spam/Ham được cải thiện.
- Loss cuối giảm từ `0.1281` xuống `0.0826`.
- F1-score tăng lên `0.8757`.

Tuy nhiên, việc tăng Vocabulary không phải lúc nào cũng làm tăng độ chính xác. Vocabulary quá lớn có thể chứa các từ ít xuất hiện, làm tăng số chiều dữ liệu, chi phí tính toán và nguy cơ học nhiễu.

### 5.4. Kết luận

Qua thí nghiệm, tôi nhận thấy kích thước Vocabulary có ảnh hưởng đến hiệu quả của mô hình BoW + Logistic Regression.

Với cùng dữ liệu và tham số huấn luyện:

- Vocabulary 100 từ đạt F1 = `0.8222`.
- Vocabulary 5.000 từ đạt F1 = `0.8757`.

Trong hai cấu hình được thử nghiệm, Vocabulary **5.000 từ cho kết quả tốt hơn**, vì mô hình có thể khai thác nhiều đặc trưng văn bản hơn.

Bài học rút ra là cần lựa chọn kích thước Vocabulary phù hợp để cân bằng giữa **độ chính xác, số chiều đặc trưng và chi phí tính toán**.

## 6. Phương án cho sản phẩm thật

### 6.1. Yêu cầu của hệ thống

Giả sử tôi cần xây dựng một hệ thống phân loại Spam/Ham cho nhà mạng với các yêu cầu:

- Xử lý **10 triệu tin nhắn SMS mỗi ngày**.
- Độ trễ p95 phải nhỏ hơn **50 ms/tin**.
- Có khả năng phân loại chính xác tin nhắn Spam và Ham.
- Đảm bảo hiệu suất ổn định khi lưu lượng tăng cao.
- Tối ưu chi phí vận hành và khả năng mở rộng.

Lưu lượng trung bình cần xử lý:

```text
10.000.000 / (24 × 60 × 60)
= 115,74 SMS/giây
≈ 116 SMS/giây
```

Như vậy, hệ thống cần xử lý trung bình khoảng **116 SMS/giây**, chưa tính các thời điểm lưu lượng tăng đột biến.

### 6.2. Phương án lựa chọn

Tôi đề xuất sử dụng **Bag-of-Words (BoW) kết hợp Logistic Regression** làm mô hình phân loại chính.

Đây là phương pháp Machine Learning truyền thống có kiến trúc đơn giản và phù hợp để triển khai cho những hệ thống cần xử lý số lượng lớn tin nhắn.

**Lý do lựa chọn:**

1. **Tốc độ xử lý:** Logistic Regression có quá trình dự đoán đơn giản, không cần thực hiện nhiều phép tính phức tạp như Deep Learning hoặc LLM.

2. **Chi phí vận hành:** Mô hình có thể chạy trực tiếp trên máy chủ, không cần gửi từng tin nhắn đến API của nhà cung cấp LLM.

3. **Khả năng mở rộng:** Có thể triển khai nhiều instance để xử lý tin nhắn song song khi lưu lượng tăng.

4. **Dễ bảo trì:** Mô hình tương đối dễ huấn luyện lại và cập nhật khi xuất hiện những dạng Spam mới.

**Số liệu cần đối chiếu từ thí nghiệm:**

| Chỉ số | BoW + Logistic Regression |
|---|---|
| F1 trên tập test | Cập nhật từ `RESULTS` |
| F1 trên SUB 40 | Cập nhật từ `RESULTS` |
| p95 (ms/tin) | Cập nhật từ `RESULTS` |
| Tốc độ (tin/giây) | Cập nhật từ `RESULTS` |
| SLA p95 < 50 ms | Xác nhận sau khi đối chiếu |

Tôi chỉ lựa chọn chính thức mô hình này nếu kết quả đo thực tế đáp ứng yêu cầu p95 dưới 50 ms/tin.

### 6.3. Đánh đổi khi sử dụng Logistic Regression

Mặc dù có ưu điểm về tốc độ và chi phí, Logistic Regression vẫn tồn tại một số hạn chế:

- Có thể phân loại sai những tin nhắn Spam được viết tự nhiên, giống tin nhắn của người quen.
- Khó nhận diện các nội dung lừa đảo tinh vi hoặc cố tình thay đổi cách viết từ khóa.
- Phụ thuộc vào chất lượng dữ liệu huấn luyện và bộ từ vựng.
- Cần cập nhật dữ liệu và huấn luyện lại khi hình thức Spam thay đổi.

Trong thí nghiệm thay đổi Vocabulary ở Phần 5, tôi thu được:

| Vocabulary | F1 test |
|---|---:|
| 100 từ | 0.8222 |
| 5.000 từ | 0.8757 |

Kết quả cho thấy sử dụng nhiều đặc trưng văn bản hơn đã cải thiện F1-score trong thí nghiệm này.

Tuy nhiên, số chiều lớn hơn cũng làm tăng chi phí tính toán và bộ nhớ. Vì vậy, cần đo lại độ trễ khi triển khai thực tế.

### 6.4. Đề xuất kiến trúc hệ thống

Tôi đề xuất xây dựng hệ thống phân loại SMS theo hai tầng.

**Tầng 1 — Logistic Regression:**

- Tiếp nhận tin nhắn SMS.
- Tiền xử lý và chuyển văn bản thành vector BoW.
- Dự đoán xác suất Spam.
- Nếu xác suất Spam lớn hơn 0.7, phân loại là Spam.
- Nếu xác suất Spam nhỏ hơn 0.3, phân loại là Ham.

**Tầng 2 — Xử lý tin nhắn khó:**

Những tin nhắn có xác suất Spam trong khoảng từ 0.3 đến 0.7 được đưa vào hàng đợi để kiểm tra bổ sung bằng mô hình Embedding hoặc LLM.

Cách tiếp cận này giúp phần lớn tin nhắn được xử lý nhanh bằng Logistic Regression, trong khi những trường hợp khó có thể được phân tích kỹ hơn.

**Luồng xử lý đề xuất:**

```text
SMS đầu vào
     |
     v
Tiền xử lý văn bản
     |
     v
BoW + Logistic Regression
     |
     v
Dự đoán xác suất Spam
     |
     +---- p > 0.7 ----> Spam
     |
     +---- p < 0.3 ----> Ham
     |
     +---- 0.3 <= p <= 0.7
                  |
                  v
             Hàng đợi
                  |
                  v
          Embedding / LLM
                  |
                  v
           Kiểm tra bổ sung
```

**Lưu ý:** Việc gọi LLM có thể vượt SLA 50 ms. Vì vậy, nếu SLA áp dụng cho toàn bộ quá trình phân loại, cần xử lý kiểm tra bổ sung bất đồng bộ và không để API LLM chặn luồng xử lý chính.

### 6.5. Ước tính chi phí nếu sử dụng LLM

Giả sử sử dụng LLM cho toàn bộ tin nhắn với chi phí theo bài tính tay:

- 1 triệu SMS: **10,50 USD**.
- 10 triệu SMS/ngày: **105 USD/ngày**.
- 30 ngày: **3.150 USD/tháng**.

Công thức:

```text
Chi phí mỗi tháng
= 10 × 30 × 10,50
= 3.150 USD
```

Đây chỉ là chi phí ước tính theo giá và lượng token giả định trong bài lab, chưa bao gồm chi phí máy chủ, mạng, retry và vận hành.

Nếu chỉ sử dụng LLM cho một tỷ lệ nhỏ các tin nhắn khó, chi phí API có thể giảm đáng kể.

### 6.6. Kết luận

Với yêu cầu xử lý 10 triệu SMS/ngày và p95 dưới 50 ms/tin, tôi ưu tiên mô hình **BoW + Logistic Regression** làm phương án chính, với điều kiện mô hình đáp ứng SLA qua kiểm thử thực tế.

Tôi chấp nhận đánh đổi một phần khả năng nhận diện ngữ nghĩa phức tạp để có tốc độ xử lý nhanh hơn, chi phí thấp hơn và khả năng mở rộng tốt hơn.

Trong tương lai, có thể kết hợp thêm Embedding hoặc LLM để phân tích các tin nhắn khó, đồng thời bổ sung dữ liệu huấn luyện mới nhằm cải thiện độ chính xác.

Tôi nhận thấy khi xây dựng sản phẩm AI thực tế, không nên chỉ lựa chọn mô hình có F1-score cao nhất mà còn phải cân nhắc **độ trễ, thông lượng, chi phí và độ ổn định**.

## 7. AI log và điều đã học

### 7.1. Nhật ký sử dụng AI (AI Log)

Trong quá trình thực hiện bài Lab 1, tôi sử dụng ChatGPT để hỗ trợ tìm hiểu kiến thức, giải thích thuật toán, kiểm tra code và xử lý những lỗi phát sinh.

Tôi ghi lại các nội dung đã sử dụng AI như sau:

| STT | Phần | Câu hỏi hoặc yêu cầu gửi AI | AI hỗ trợ những gì? | Tôi đã thực hiện và kiểm chứng |
|---|---|---|---|---|
| 1 | Machine Learning | Giải thích cách xây dựng mô hình Bag-of-Words và Logistic Regression | Giải thích cách vector hóa văn bản, hàm Sigmoid, Gradient Descent và F1-score | Chạy các cell Phần 1, kiểm tra quá trình huấn luyện và kết quả đánh giá |
| 2 | Deep Learning | Kiểm tra code xây dựng mạng MLP bằng PyTorch | Giải thích kiến trúc Linear → ReLU → Linear và cách sử dụng BCEWithLogitsLoss | Kiểm tra hàm `build_mlp()`, chạy mô hình và kiểm tra kích thước đầu ra |
| 3 | Embedding | Hướng dẫn xử lý lỗi biến và giải thích Sentence Embedding | Giải thích vector ngữ nghĩa, Cosine Similarity và kNN | Chạy lại các cell khởi tạo Embedding và kiểm tra kết quả |
| 4 | LLM | Hướng dẫn viết prompt và xử lý kết quả phân loại Spam/Ham | Giải thích Zero-shot, Few-shot, `parse_label()` và cách gọi API | Chạy thử nghiệm trên tập SUB gồm 40 tin nhắn |
| 5 | LLM API | Hỏi cách xử lý lỗi 429, 503 và các lỗi `NameError` | Hướng dẫn import thư viện, cấu hình provider và xử lý retry | Chỉnh sửa code và chạy lại những cell bị lỗi |
| 6 | Groq API | Hỏi cách chuyển từ Gemini sang Groq | Hướng dẫn tạo API Key, lưu Colab Secrets và cấu hình OpenAI-compatible endpoint | Chuyển provider sang Groq và tiếp tục thí nghiệm |
| 7 | Few-shot | Hỏi cách thực hiện thí nghiệm Few-shot | Hướng dẫn bổ sung 4 ví dụ vào prompt và so sánh với Zero-shot | Chạy 40 tin nhắn, thu được F1 = 0.952 |
| 8 | Phân tích lỗi | Hỏi vì sao `parse_label_v2()` phân loại sai | Giải thích lỗi kiểm tra chuỗi con và cách chuẩn hóa nhãn | Kiểm tra các trường hợp như `"Not spam"` và phản hồi không hợp lệ |
| 9 | Phần 5 | Hỏi cách đọc biểu đồ Pareto, SLA và stress test | Giải thích quan hệ giữa F1, độ trễ, chi phí và các nhóm case khó | Đối chiếu kết quả kiểm thử và phân tích những tin nhắn dễ gây nhầm lẫn |
| 10 | Tính tay | Hỏi cách tính Accuracy, Sigmoid, F1, Cosine Similarity và chi phí LLM | Hướng dẫn công thức và từng bước tính | Kiểm tra phép tính, đối chiếu số lượng 5.169 SMS sau khi bỏ trùng |
| 11 | Báo cáo | Hỏi cách trình bày kết quả và viết file Markdown trên GitHub | Hướng dẫn định dạng bảng, tiêu đề và cách tổ chức nội dung báo cáo | Bổ sung số liệu thực nghiệm và chỉnh sửa file `.md` |

### 7.2. Những lỗi đã gặp và cách khắc phục

Trong quá trình thực hiện bài lab, tôi gặp một số lỗi tiêu biểu:

**Lỗi 1: `NameError` trong Python**

- Nguyên nhân: Một số thư viện hoặc hàm chưa được import, định nghĩa hoặc chạy trong phiên Google Colab hiện tại.
- Cách khắc phục: Bổ sung các thư viện như `os`, `time`, `RateLimitError`, `InternalServerError` và chạy các cell theo đúng thứ tự.
- Bài học: Trong Jupyter Notebook, các biến và hàm chỉ tồn tại khi cell tương ứng đã được thực thi trong Runtime.

**Lỗi 2: Gemini API trả về 429 và 503**

- Lỗi 429 xuất hiện khi vượt giới hạn request hoặc quota.
- Lỗi 503 xuất hiện khi dịch vụ hoặc mô hình đang quá tải.
- Cách xử lý: Sử dụng retry có giới hạn, exponential backoff và chuyển sang nhà cung cấp Groq để tiếp tục thử nghiệm.
- Bài học: Khi triển khai LLM, cần quan tâm đến quota, tốc độ phản hồi và độ ổn định của API.

**Lỗi 3: Gọi sai hàm vector hóa**

- Ban đầu tôi sử dụng `vectorize_bow()` và gặp lỗi `NameError`.
- Sau khi kiểm tra notebook, tôi sử dụng đúng hàm `vectorize()`.
- Bài học: Cần kiểm tra tên hàm, tham số đầu vào và các giá trị trả về trước khi sử dụng.

**Lỗi 4: Sai định dạng Markdown**

- Khi viết câu trả lời trong Google Colab, danh sách đánh số tự động khiến nội dung bị hiển thị sai.
- Tôi khắc phục bằng cách sử dụng tiêu đề và định dạng in đậm như `**Câu 1:**`, `**Câu 2:**`.
- Bài học: Cần kiểm tra cách hiển thị Markdown sau khi chỉnh sửa nội dung.

### 7.3. Điều tôi học được sau bài Lab

Trước khi thực hiện bài lab, tôi nghĩ rằng mô hình AI càng lớn hoặc được cung cấp càng nhiều ví dụ thì kết quả phân loại sẽ càng chính xác. Tuy nhiên, kết quả thử nghiệm cho thấy Gemini nhỏ Zero-shot đạt F1 = 1.000 trên 40 tin nhắn, trong khi Few-shot đạt F1 = 0.952 khi sử dụng nhà cung cấp khác. Điều này giúp tôi hiểu rằng cần đánh giá các phương pháp trên cùng mô hình và dữ liệu trước khi đưa ra kết luận.

Tôi cũng hiểu rõ hơn sự khác biệt giữa Machine Learning, Deep Learning, Embedding và LLM. Đặc biệt, trong thử nghiệm thay đổi Vocabulary, F1-score của Logistic Regression tăng từ 0.8222 lên 0.8757 khi tăng số lượng từ từ 100 lên 5.000. Qua đó, tôi nhận thấy việc lựa chọn đặc trưng đầu vào có ảnh hưởng lớn đến chất lượng mô hình.

Ngoài độ chính xác, tôi học được rằng một hệ thống AI thực tế còn phải cân nhắc độ trễ, thông lượng, chi phí, khả năng mở rộng và độ ổn định. AI có thể hỗ trợ giải thích kiến thức và xử lý lỗi, nhưng người lập trình cần tự chạy thử, kiểm tra số liệu và hiểu nguyên nhân của kết quả.

### 7.4. Kết luận

Qua bài Lab 1, tôi đã có cơ hội thực hành toàn bộ quy trình xây dựng hệ thống phân loại Spam/Ham, từ xử lý dữ liệu, xây dựng mô hình Machine Learning, Deep Learning và Embedding đến sử dụng LLM API.

Việc sử dụng ChatGPT giúp tôi hiểu nhanh hơn những kiến thức chưa rõ và khắc phục các lỗi trong quá trình thực hành. Tuy nhiên, tôi nhận thấy AI chỉ nên đóng vai trò công cụ hỗ trợ; việc kiểm chứng, đánh giá và chịu trách nhiệm về kết quả vẫn thuộc về người thực hiện.

Bài lab giúp tôi hiểu rằng lựa chọn mô hình AI không chỉ dựa trên F1-score mà còn phải phù hợp với yêu cầu kỹ thuật và điều kiện triển khai thực tế.
