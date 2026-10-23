# ⚡ PayPilot AI — Intelligent Payroll & Workforce Management System

An enterprise-ready, AI-assisted Payroll and HR Management System built with **Python 3, Flask, SQLite, Scikit-Learn (Random Forest ML), Chart.js**, and an **Intelligent HR & Tax Chatbot**.

---

## 🚀 Quick Start (One-Click Run / நேரடி இயக்கம்)

### Option 1: Double-click Windows Batch File (Recommended)
Simply double-click:
```
run_windows.bat
```
This automatically sets up Python virtual environment `.venv`, verifies all packages, initializes the database with demo records, and opens your browser directly to `http://127.0.0.1:5000`!

### Option 2: PowerShell One-Click Launcher
```powershell
.\run_app.ps1
```

### Option 3: Manual Terminal Command
```powershell
.\.venv\Scripts\activate.bat
python app.py
```

---

## 🔑 Default Admin Credentials (உள்நுழைவு விவரங்கள்)

| Field | Default Value | Note |
| :--- | :--- | :--- |
| **URL** | `http://127.0.0.1:5000` | Local Web Server |
| **Username** | `admin` | Default Admin Account |
| **Password** | `admin123` | Configured in `.env` |
| **Quick Login** | *1-Click Demo Button* | Available directly on the Login page |

---

## ✨ Features Overview (புதிய வசதிகள் & அம்சங்கள்)

### 1. 📊 Executive Interactive Analytics Dashboard
- **Chart.js Visualizations**:
  - **Monthly Payroll Outflow**: Visual area/bar comparison of Gross vs Net pay.
  - **Department Salary Distribution**: Doughnut breakdown across Engineering, Analytics, HR, Finance, QA.
- **KPI Summary Cards**: Active employees, total payslips issued, disbursed funds, and compliance audit risk level.
- **Recent Activity Ledger**: Fast access to recent salary runs and employee slips.

### 2. 📅 Monthly Attendance & Leave Management (NEW!)
- Track monthly **Working Days, Days Present, Paid Leaves, Unpaid Leaves (Loss of Pay - LOP)**, and **Overtime Hours**.
- **Automated Loss of Pay (LOP) Deduction**:
  $$\text{Daily Rate} = \frac{\text{Monthly Basic}}{\text{Total Working Days}}$$
  $$\text{LOP Deduction} = \text{Unpaid Leaves} \times \text{Daily Rate}$$
- **Overtime Bonus**: Automatically credits overtime at 1.5x regular hourly compensation.
- **1-Click Auto-fill Full Attendance (22 Days)**: Pre-fills full present days for all employees in one click.

### 3. ⚡ 1-Click Automated Bulk Payroll Run (NEW!)
- Instead of manually entering each employee one-by-one, click **"Run Payroll for All Active Employees"**.
- Automatically computes:
  - 15% HRA / Special Allowance
  - 12% Statutory EPF (Employee Provident Fund)
  - Synced Attendance LOP deductions and Overtime bonuses.

### 4. 📄 Professional Corporate Salary Slip (Print / Save as PDF)
- Clean, corporate payslip invoice template complete with company seal, digital signature block, detailed earnings vs deductions split, and watermark.
- Optimized for standard A4 paper printing (`window.print()` / Save as PDF).

### 5. 🤖 Machine Learning Salary Predictor (Random Forest Regressor)
- Predicts fair market compensation using **Scikit-Learn Random Forest Regression** trained on real benchmarks.
- Features: Years of experience, Highest education (UG/PG/Diploma), Designation, and Performance rating (1–5).
- Outputs expected annual salary, monthly take-home, and market confidence ranges.
- **Cached in-memory model**: Instant sub-millisecond predictions without retraining on every click.

### 6. 📈 AI Appraisal & Salary Increment Recommender (NEW!)
- Evaluates employee performance ratings (1 to 5), tenure, and market parity gap.
- Recommends an optimal **% salary hike** (e.g. +14.5% hike) along with an automated **AI Rationale Justification** for HR committee reviews.

### 7. 🔍 AI Payroll Anomaly & Fraud Audit Scanner (NEW!)
- Real-time compliance scanner that checks every payroll record for:
  - Negative or zero net take-home salaries
  - Excessive deduction ratios (>40% of gross pay)
  - Abnormal bonus spikes (>50% of basic salary)
  - Gross calculation discrepancies
- Displays real-time **Risk Severity (High / Medium / Low)** and compliance score.

### 8. 💬 Intelligent HR & Tax Chatbot Assistant
- Dual-mode operation:
  - **Offline Knowledge Engine**: 30+ comprehensive HR rules (PF, ESI, HRA, Form 16, TDS, Gratuity, LOP, CTC).
  - **OpenAI LLM Integration**: Connects to `gpt-4o-mini` if `OPENAI_API_KEY` is placed in `.env`.
- Interactive suggested prompt chips for 1-click asking.

### 9. 📥 Instant CSV Export Reports (NEW!)
- Export Complete Payroll Ledger to CSV (`/payroll/export`).
- Export Employee Master Directory to CSV (`/employees/export`).

---

## 📁 Project Structure

```
ai_payroll_management_system/
├── app.py                   # Main Flask application & ML/AI engines
├── requirements.txt         # Dependencies (Flask, SQLAlchemy, scikit-learn, pandas)
├── run_windows.bat          # 1-Click Windows Batch launcher
├── run_app.ps1              # 1-Click PowerShell launcher
├── .env                     # Local configuration & credentials
├── .env.example             # Example configuration template
├── data/
│   └── salary_training.csv  # Synthetic compensation training dataset
├── instance/
│   └── payroll.db           # SQLite database (auto-created & seeded)
├── templates/
│   ├── base.html            # Navigation, layout, Bootstrap 5.3 & Chart.js
│   ├── index.html           # Executive dashboard & interactive graphs
│   ├── employees.html       # Employee directory with search & filters
│   ├── employee_form.html   # Add/Edit employee with live CTC preview
│   ├── attendance.html      # Attendance & Loss of Pay (LOP) tracker
│   ├── payroll.html         # 1-click bulk & manual payroll runs
│   ├── payslip.html         # Corporate printable salary slip
│   ├── prediction.html      # Machine learning salary predictor
│   ├── appraisal.html       # AI appraisal & increment recommender
│   ├── anomalies.html       # AI payroll anomaly & fraud detector
│   ├── chatbot.html         # Intelligent HR AI assistant
│   └── login.html           # Admin sign-in with 1-click demo autofill
└── static/
    ├── css/style.css        # Modern glassmorphism & responsive stylesheet
    └── js/main.js           # Client-side scripts & auto-dismiss alerts
```

---

## 🛠️ Technology Stack

- **Backend**: Python 3.13, Flask 3.1, Flask-SQLAlchemy, Flask-Login, Werkzeug
- **Database**: SQLite3 (Local, zero external server setup required)
- **Machine Learning**: Scikit-Learn (Random Forest Regressor, Pipeline, ColumnTransformer, OneHotEncoder), Pandas, NumPy
- **Frontend**: HTML5, Vanilla CSS3 (Custom Design System), Bootstrap 5.3, Bootstrap Icons, Chart.js 4.4
- **AI / LLM**: Optional OpenAI API integration with built-in comprehensive offline fallback
