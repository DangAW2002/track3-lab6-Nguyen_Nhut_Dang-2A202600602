# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Nhựt Đăng — 2A202600602
**Ngày nộp**: 2026-06-25
**Submission option**: Option A (Lightweight ZIP) / Option B (HF Hub) / Option C (Code-only)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (200 samples: 180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 562, rounded up to power of 2)
- **GPU**: Tesla T4 (16 GB VRAM) on Google Colab
- **Training cost**: $0.07 (Tổng 11.5 phút @ $0.35/hr)
- **HF Hub link** (nếu chọn Option B): https://huggingface.co/AWDang/qwen2.5-3b-vi-lab21-r16

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM  | Eval Loss | Perplexity |
|------|-----------------|------------|------------|-----------|------------|
| 8    | 1,843,200       | 3.85 min   | 13.08 GB   | 1.5577    | 4.75       |
| 16   | 3,686,400       | 3.80 min   | 12.48 GB   | 1.5161    | 4.55       |
| 64   | 14,745,600      | 3.85 min   | 13.86 GB   | 1.4768    | 4.38       |
| Base | -               | -          | -          | N/A       | N/A        |


## 3. Loss Curve Analysis
*(Hình ảnh loss_curve.png được đính kèm ở thư mục results/)*
- **Quan sát**: Quá trình huấn luyện diễn ra mượt mà và tổn hao (training loss) giảm dần theo thời gian. Mặc dù eval-during-training bị tắt để tránh tràn VRAM trên GPU T4, việc so sánh perplexity của các rank cho thấy rank cao hơn (`r=64`) có độ nhiễu thấp hơn (loss thấp nhất: 1.4768) mà không bị overfitting quá mức do số lượng mẫu huấn luyện còn tương đối nhỏ (180 mẫu) và thời gian huấn luyện ngắn.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, n...
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân ...
**Nhận xét**: Improved. Bản fine-tuned đưa ra định nghĩa rõ ràng hơn, nhấn mạnh yếu tố "tự học từ dữ liệu mà không cần lập trình trực tiếp" (không có sự hướng dẫn trực tiếp từ người dùng).

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:
```python
def fibonacci(n):
    if n <= 0:
        return "N p...
```
**Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:
```python
def fibonacci(n):
    if n < 0:
        raise ValueError("Input phải là một số nguyên dương.")
    elif n == 0:
   ...
```
**Nhận xét**: Improved. Bản fine-tuned bổ sung kiểm tra điều kiện đầu vào chặt chẽ hơn (`ValueError` khi `n < 0`), giúp code an toàn và chuyên nghiệp hơn.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc,...
**Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX th...
**Nhận xét**: Improved. Bản fine-tuned liệt kê các nguyên tắc thiết thực và mang tính kỹ thuật/chuyên nghiệp hơn (Chuyển đổi, Thích ứng, Đơn giản) thay vì định nghĩa chung chung lặp từ của bản base.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hi...
**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong...
**Nhận xét**: Same/Slightly Degraded (về định nghĩa viết tắt). Bản base nêu đúng bản chất LoRA là "Low-Rank Adaptation", trong khi bản fine-tuned định nghĩa LoRA thành "Layer-wise Adaptive Regularization Optimization" (đây là một lỗi hallucination nhỏ về tên viết tắt). Tuy nhiên, cả hai đều thể hiện được bối cảnh tối ưu hiệu năng.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của ...
**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp ...
**Nhận xét**: Improved. Bản fine-tuned đưa ra câu trả lời gọn gàng, định nghĩa chính xác và phân loại rõ ràng mục đích của từng kỹ thuật trong hệ sinh thái AI.

## 5. Conclusion về Rank Trade-off
Dựa trên kết quả thực nghiệm huấn luyện Qwen2.5-3B, cấu hình rank **r=16** (alpha=32) mang lại tỷ lệ ROI (tỷ lệ hiệu quả trên chi phí) tốt nhất trên tập dữ liệu này. Khi tăng rank từ r=8 lên r=16, độ phức tạp (perplexity) giảm rõ rệt từ 4.75 xuống 4.55 (chênh lệch 0.20), đồng thời lượng VRAM tiêu thụ thực tế lại giảm xuống mức tối ưu nhất là 12.48 GB (thấp hơn cả r=8 do tính tối ưu của các kernel phân bổ bộ nhớ). Khi tăng tiếp rank lên r=64 (gấp 4 lần số lượng tham số trainable từ 3.69M lên 14.75M), perplexity chỉ giảm thêm 0.17 xuống 4.38, cho thấy rõ hiện tượng **Diminishing Returns** (hiệu quả tăng chậm lại) trong khi lượng VRAM tiêu thụ tăng vọt lên mức cao nhất là 13.86 GB. Vì vậy, nếu deploy production cho use case này, cấu hình rank **r=16** là lựa chọn tối ưu nhất để cân bằng giữa chất lượng phản hồi và chi phí tài nguyên.


## 6. What I Learned
- Hiểu được cơ chế hoạt động thực tế của LoRA/QLoRA và cách dùng Unsloth để tăng tốc độ huấn luyện mô hình (giúp giảm đáng kể VRAM tiêu thụ và thời gian chạy).
- Nhận diện được sự đánh đổi (trade-off) giữa việc tăng rank (r) và chi phí tài nguyên (VRAM, số lượng tham số huấn luyện, thời gian huấn luyện).
- Biết cách chuẩn bị và format dataset theo cấu trúc Alpaca chuẩn để huấn luyện các mô hình Instruct.
