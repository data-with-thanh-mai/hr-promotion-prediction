# HR Promotion Prediction API
## 1. Trải nghiệm Giao diện UI (Interactive API Demo)
## 2. Business Context
A large MNC have 9 broad verticals across the organisation. One of the problem is identifying the right people for promotion (only for manager position and below) and prepare them in time.

The final promotions are only announced after the evaluation and this leads to delay in transition to new roles. Hence, company needs help in identifying the eligible candidates at a particular checkpoint so that they can expedite the entire promotion cycle.

Multiple attributes have been provided around Employee's past and current performance along with demographics.
## 3. Nguồn Dữ liệu & Tiền xử lý (Data & Preprocessing)
There are 54808 samples, 13 features. 
Data description 
- employee_id: Unique ID for employee
- department: Department of employee
- region: Region of employment (unordered)
- education: Education Level
- gender: Gender of Employee
- recruitment_channel: Channel of recruitment for employee
- no_ of_ trainings: no of other trainings completed in previous year on soft skills, technical skills etc.
- age: Age of Employee
- previous_ year_ rating: Employee Rating for the previous year
- length_ of_ service: Length of service in years
- awards_ won?: if awards won during previous year then 1 else 0
- avg_ training_ score: Average score in current training evaluations
- is_promoted: (Target) Recommended for promotion

- Đặc diểm của bộ data:
  Dữ liệu bị mất cân bằng :  class 1 chiếm khoảng 8.69% ( 3188 samples) trong tập dữ liệu -> giải quyết băngg class_weight( ko dùng smore vì data có categorical sẽ sinh ra những điểm dữ liệu không hợp lí)
- Mising : ở hai cột previouse_year_rating và education
    - Previouse_year_rating: nan là do đây là những nhân sự mới làm năm đầu tiên nên không có dữ liệu chấm điểm ở năm trước -> điền 0 ( chiếm khoảng 7.63%),
    - education: chiếm khảong 4.38% -> quyết định fill most frequnet -> ko chiếm quá nhiều trong tập dữ liệu gây nhiễu model 
- Loại bỏ 2 cột : gender, age và employee_id ( đinh danh ko có ý nghĩa về mặt dự đoán)
    - cả gender và age đều ko cho thấy giá trị trong việc giúp model đưa ra dự đoán ( đặt biệt là age thông qua kiểu đinh với p-value = 0.05)
    - Viêkc loại bỏ 2 yếu tố này cũng giảm bớt tính thiên vị do giới tính và tuoiir -> giúp lựa chọn nhân sự dựa trên năng lực của họ thay vì tuổi tác hay giới tính
- Các feature còn lại : có thấy những giá trị đóng góp ít nhiều:
-   - Education : odinary mapping vì nhận thấy ngta có trình độ học vấn cao hơn thường có xu hướng đưojc cắt nhắt nhiều hơn
- Về engnireing : tạo thêm hai cái feature mới :
-   - relative_ training_score: tỷ lệ của nhân sự so với trung bình của phòng ban đó
    - total_training_score  : ave * no_traning ->
   
## 4. Kiến trúc Mô hình & Ngưỡng quyết định (Model Architecture & Threshold Strategy)

## 5. Cấu trúc Thư mục Dự án (Repository Structure

## 6. Yêu cầu Hệ thống & Cài đặt (Setup & Installation)

## 7. Hướng dẫn Khởi chạy Server (Running the API)
## 8. Hướng dẫn Tương tác API (API Usage & Examples)
## 9. Lộ trình Phát triển Tiếp theo (Future Enhancements)
