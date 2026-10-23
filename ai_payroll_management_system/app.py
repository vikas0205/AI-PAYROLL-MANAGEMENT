import csv
import io
import math
import os
from datetime import date, datetime
from functools import wraps

import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask, flash, jsonify, redirect, render_template, request,
    send_file, url_for
)
from flask_login import (
    LoginManager, UserMixin, current_user, login_required, login_user, logout_user
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()

app = Flask(__name__, instance_relative_config=True)
os.makedirs(app.instance_path, exist_ok=True)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "paypilot-ai-super-secret-key-2026-secure")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(app.instance_path, "payroll.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

# Global cached ML model for instant prediction
_CACHED_SALARY_MODEL = None


# ---------------------------------------------------------
# Database Models
# ---------------------------------------------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(30), nullable=False, default="admin")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_code = db.Column(db.String(30), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(160), nullable=False)
    phone = db.Column(db.String(30), nullable=True, default="+91 9876543210")
    department = db.Column(db.String(80), nullable=False)
    designation = db.Column(db.String(100), nullable=False)
    experience_years = db.Column(db.Integer, nullable=False, default=0)
    education = db.Column(db.String(20), nullable=False, default="UG")
    performance_rating = db.Column(db.Integer, nullable=False, default=3)
    monthly_basic = db.Column(db.Float, nullable=False, default=25000.0)
    active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    attendances = db.relationship("Attendance", backref="employee", cascade="all, delete-orphan", lazy=True)
    payrolls = db.relationship("Payroll", backref="employee", cascade="all, delete-orphan", lazy=True)


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employee.id"), nullable=False)
    month = db.Column(db.String(7), nullable=False)  # YYYY-MM
    total_working_days = db.Column(db.Integer, nullable=False, default=22)
    present_days = db.Column(db.Integer, nullable=False, default=22)
    paid_leaves = db.Column(db.Integer, nullable=False, default=0)
    unpaid_leaves = db.Column(db.Integer, nullable=False, default=0)
    overtime_hours = db.Column(db.Float, nullable=False, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("employee_id", "month", name="uq_emp_month_attendance"),
    )

    @property
    def loss_of_pay_deduction(self):
        if self.employee and self.total_working_days > 0:
            daily_rate = self.employee.monthly_basic / self.total_working_days
            return round(self.unpaid_leaves * daily_rate, 2)
        return 0.0

    @property
    def overtime_pay(self):
        if self.employee and self.total_working_days > 0 and self.overtime_hours > 0:
            hourly_rate = (self.employee.monthly_basic / self.total_working_days) / 8.0
            return round(self.overtime_hours * hourly_rate * 1.5, 2)
        return 0.0


class Payroll(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employee.id"), nullable=False)
    pay_month = db.Column(db.String(7), nullable=False)  # YYYY-MM
    basic = db.Column(db.Float, nullable=False)
    allowance = db.Column(db.Float, nullable=False, default=0.0)
    bonus = db.Column(db.Float, nullable=False, default=0.0)
    deductions = db.Column(db.Float, nullable=False, default=0.0)
    gross = db.Column(db.Float, nullable=False)
    net = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False, default="Paid")
    notes = db.Column(db.String(255), nullable=True, default="Processed via PayPilot AI")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("employee_id", "pay_month", name="uq_emp_month_payroll"),
    )


# ---------------------------------------------------------
# Login & Permissions Helper
# ---------------------------------------------------------

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_globals():
    return {
        "today": date.today(),
        "currency": "₹",
        "current_month_str": date.today().strftime("%Y-%m"),
    }


# ---------------------------------------------------------
# AI & Machine Learning Salary Prediction Engine
# ---------------------------------------------------------

def get_or_train_salary_model():
    """Trains or returns the cached Random Forest model for salary prediction."""
    global _CACHED_SALARY_MODEL
    if _CACHED_SALARY_MODEL is not None:
        return _CACHED_SALARY_MODEL

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        csv_path = os.path.join(app.root_path, "data", "salary_training.csv")
        data = pd.read_csv(csv_path)
        features = ["experience_years", "education", "designation", "performance_rating"]
        categorical = ["education", "designation"]
        numeric = ["experience_years", "performance_rating"]

        preprocessor = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ])
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ])
        model.fit(data[features], data["annual_salary"])
        _CACHED_SALARY_MODEL = model
        return _CACHED_SALARY_MODEL
    except Exception as exc:
        app.logger.warning(f"Failed to train scikit-learn model: {exc}. Using heuristic fallback.")
        return None


def predict_annual_salary(experience, education, designation, rating):
    """Predict annual salary using ML model with a reliable algorithmic fallback."""
    model = get_or_train_salary_model()
    if model is not None:
        try:
            frame = pd.DataFrame([{
                "experience_years": experience,
                "education": education,
                "designation": designation,
                "performance_rating": rating,
            }])
            pred = float(model.predict(frame)[0])
            return max(240000.0, pred)
        except Exception as exc:
            app.logger.warning(f"Model prediction exception: {exc}")

    # Robust algorithmic fallback
    base_salaries = {
        "Junior Developer": 320000,
        "Python Developer": 520000,
        "Frontend Developer": 480000,
        "Full Stack Developer": 750000,
        "Senior Developer": 980000,
        "Team Lead": 1300000,
        "Data Analyst": 500000,
        "Data Scientist": 900000,
        "QA Engineer": 420000,
        "Senior QA Engineer": 720000,
        "HR Executive": 360000,
        "HR Manager": 750000,
        "Accountant": 400000,
        "Finance Manager": 800000,
    }
    base = base_salaries.get(designation, 450000)
    exp_factor = 1.0 + (experience * 0.12)
    edu_factor = 1.2 if education == "PG" else (1.0 if education == "UG" else 0.85)
    rating_factor = 0.85 + (rating * 0.05)
    return max(240000.0, base * exp_factor * edu_factor * rating_factor)


def calculate_ai_appraisal(employee, current_salary, rating, market_salary):
    """Calculates AI-recommended salary increment percentage and rationale."""
    # Base hike based on rating
    rating_hikes = {1: 0.0, 2: 3.0, 3: 8.0, 4: 14.0, 5: 22.0}
    base_hike = rating_hikes.get(rating, 8.0)

    # Market gap adjustment
    ratio = current_salary / market_salary if market_salary > 0 else 1.0
    if ratio < 0.8:
        # Underpaid compared to market: boost increment by up to 5%
        market_boost = 5.0
        gap_note = "Employee is currently below competitive market benchmark; adjustment applied."
    elif ratio > 1.2:
        market_boost = -2.0
        gap_note = "Employee is already at upper market bracket."
    else:
        market_boost = 1.0
        gap_note = "Salary is well aligned with market benchmark."

    # Experience bonus
    exp_boost = min(3.0, employee.experience_years * 0.5)

    final_hike_pct = max(0.0, min(35.0, base_hike + market_boost + exp_boost))
    new_monthly = round(current_salary * (1 + (final_hike_pct / 100.0)), 2)
    new_annual = round(new_monthly * 12, 2)

    return {
        "hike_percentage": round(final_hike_pct, 1),
        "current_monthly": current_salary,
        "new_monthly": new_monthly,
        "new_annual": new_annual,
        "market_benchmark": round(market_salary / 12, 2),
        "gap_note": gap_note,
        "rating_level": "Outstanding" if rating == 5 else ("Exceeds Expectations" if rating == 4 else ("Meets Expectations" if rating == 3 else "Needs Improvement")),
    }


# ---------------------------------------------------------
# AI Anomaly Detection Engine
# ---------------------------------------------------------

def detect_payroll_anomalies(payrolls):
    """Detects unusual salary spikes, deduction mismatches, and outlier records."""
    if not payrolls:
        return {"anomalies": [], "total_checked": 0, "risk_score": 0, "status": "No Records"}

    records = []
    anomalies = []
    for p in payrolls:
        issues = []
        # Check 1: Negative or zero net pay
        if p.net <= 0:
            issues.append("Net salary is zero or negative.")

        # Check 2: High deductions ratio (>40% of gross)
        if p.gross > 0 and (p.deductions / p.gross) > 0.40:
            issues.append(f"High deductions ({round((p.deductions/p.gross)*100, 1)}% of gross pay).")

        # Check 3: Abnormal Bonus (> 50% of monthly basic)
        if p.basic > 0 and (p.bonus / p.basic) > 0.50:
            issues.append(f"Unusually large bonus ({round((p.bonus/p.basic)*100)}% of basic).")

        # Check 4: Gross mismatch with basic + allowance + bonus
        expected_gross = round(p.basic + p.allowance + p.bonus, 2)
        if abs(p.gross - expected_gross) > 1.0:
            issues.append("Gross pay calculation discrepancy detected.")

        if issues:
            anomalies.append({
                "payroll_id": p.id,
                "employee_code": p.employee.employee_code,
                "name": p.employee.name,
                "department": p.employee.department,
                "month": p.pay_month,
                "gross": p.gross,
                "net": p.net,
                "issues": issues,
                "severity": "High" if len(issues) > 1 or p.net <= 0 else "Medium",
            })

    risk_score = min(100, int((len(anomalies) / max(1, len(payrolls))) * 100))
    status = "Low Risk" if risk_score < 15 else ("Moderate Risk" if risk_score < 40 else "High Risk")

    return {
        "anomalies": anomalies,
        "total_checked": len(payrolls),
        "anomalous_count": len(anomalies),
        "risk_score": risk_score,
        "status": status,
    }


# ---------------------------------------------------------
# AI HR Chatbot Knowledge Base
# ---------------------------------------------------------

def chatbot_reply(question):
    """Use the OpenAI API when configured; otherwise provide rich offline knowledge."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.3,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are PayPilot AI, an expert corporate HR & Payroll specialist. "
                            "Explain Indian and global payroll concepts, PF, ESI, gratuity, TDS/tax deductions, "
                            "gross vs net salary, allowances (HRA, DA, Special Allowance), attendance/loss of pay (LOP), "
                            "and performance appraisal clearly with concrete examples and friendly tone. "
                            "Keep answers structured with bullet points where helpful."
                        ),
                    },
                    {"role": "user", "content": question[:2000]},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            app.logger.exception("LLM chatbot request failed")

    # Comprehensive offline knowledge engine
    q = question.lower()

    if any(k in q for k in ["gross salary", "gross pay"]):
        return (
            "📌 **Gross Salary Explanation**:\n"
            "Gross salary is the total remuneration an employee earns before any statutory or voluntary deductions.\n\n"
            "• **Formula**: Gross Pay = Basic Salary + HRA + Special Allowances + Overtime + Bonuses\n"
            "• It does not factor in PF, Professional Tax, TDS (Tax Deducted at Source), or Loss of Pay."
        )

    if any(k in q for k in ["net salary", "take home", "take-home"]):
        return (
            "💰 **Net Salary (Take-Home Pay)**:\n"
            "Net salary is the actual amount credited to the employee's bank account every month.\n\n"
            "• **Formula**: Net Pay = Gross Salary − Total Deductions\n"
            "• Total Deductions include: Employee Provident Fund (PF), ESI, Professional Tax (PT), Income Tax (TDS), and Loss of Pay (LOP)."
        )

    if any(k in q for k in ["pf", "provident fund", "epf"]):
        return (
            "🏛️ **Provident Fund (EPF) Rules**:\n"
            "• **Employee Contribution**: 12% of (Basic Salary + DA).\n"
            "• **Employer Contribution**: 12% matched (3.67% goes to EPF, 8.33% goes to EPS pension scheme).\n"
            "• Mandated for companies with 20+ employees. Interest is set by the EPFO annually."
        )

    if any(k in q for k in ["hra", "house rent"]):
        return (
            "🏠 **House Rent Allowance (HRA)**:\n"
            "HRA helps employees pay for residential rental accommodation. It is eligible for tax exemption under Section 10(13A) of Income Tax Act.\n"
            "Exemption is the lowest of:\n"
            "1. Actual HRA received\n"
            "2. 50% of Basic (Metro cities) or 40% of Basic (Non-metro)\n"
            "3. Rent paid minus 10% of Basic salary."
        )

    if any(k in q for k in ["attendance", "lop", "loss of pay", "unpaid leave"]):
        return (
            "📅 **Attendance & Loss of Pay (LOP)**:\n"
            "• **Daily Rate**: Monthly Basic Salary ÷ Total Working Days in Month (e.g. ₹44,000 ÷ 22 = ₹2,000/day).\n"
            "• **LOP Deduction**: Unpaid Leave Days × Daily Rate.\n"
            "• PayPilot AI includes an Attendance module that automatically calculates LOP deductions and Overtime allowance when generating monthly payroll!"
        )

    if any(k in q for k in ["overtime", "ot"]):
        return (
            "⏱️ **Overtime (OT) Calculation**:\n"
            "• **Hourly Rate**: Daily Rate ÷ 8 working hours.\n"
            "• As per the Factories & Shops Act standard, overtime is compensated at 1.5x to 2x the standard hourly rate.\n"
            "• PayPilot automatically adds Overtime bonus to the employee's gross pay."
        )

    if any(k in q for k in ["appraisal", "increment", "hike", "rating"]):
        return (
            "📈 **AI Appraisal & Salary Hike Policy**:\n"
            "PayPilot AI uses a 3-pillar appraisal engine:\n"
            "1. **Performance Rating (1 to 5)**: Rating 5 (18–25% hike), Rating 4 (12–16%), Rating 3 (6–10%), Rating 1-2 (0–3%).\n"
            "2. **Market Gap Analysis**: Adjusts hike higher if the employee is paid below market peers.\n"
            "3. **Experience Tenure**: Extra loyalty boost based on years of proven delivery."
        )

    if any(k in q for k in ["payslip", "salary slip", "pdf"]):
        return (
            "📄 **Payslip Information**:\n"
            "A Payslip is an official statement issued by the employer. In PayPilot AI, you can view the salary slip for any employee and click 'Print / Save PDF' to print or download a clean corporate format with company seal."
        )

    if any(k in q for k in ["anomaly", "fraud", "audit"]):
        return (
            "🔍 **AI Anomaly & Discrepancy Detection**:\n"
            "PayPilot AI features an automated audit scanner that inspects all payroll runs for:\n"
            "• Disproportionate bonuses (>50% of basic)\n"
            "• Unusually high deductions (>40% of gross)\n"
            "• Zero or negative net salaries\n"
            "• Missing payroll runs for active employees."
        )

    if any(k in q for k in ["ctc", "cost to company"]):
        return (
            "💼 **CTC (Cost to Company)**:\n"
            "CTC is the complete annual expense the company spends on an employee.\n"
            "• **CTC** = Gross Salary + Employer PF + Gratuity + Health Insurance + Variable Bonus + Perks.\n"
            "• Take-Home salary is typically 70%–85% of CTC depending on tax slabs."
        )

    if any(k in q for k in ["tax", "tds", "income tax"]):
        return (
            "📊 **TDS (Tax Deducted at Source)**:\n"
            "Employers deduct TDS monthly based on the employee's estimated annual taxable income under either the Old Tax Regime (with 80C, 80D, HRA deductions) or the New Concessional Tax Regime (simplified slabs with standard ₹75,000 deduction)."
        )

    return (
        "👋 **Welcome to PayPilot AI Assistant!**\n\n"
        "I can answer questions on:\n"
        "• **Gross vs Net Salary** & Take-home calculation\n"
        "• **Provident Fund (PF)**, ESI, Gratuity & Professional Tax\n"
        "• **HRA** calculation & tax exemption rules\n"
        "• **Attendance & Loss of Pay (LOP)** deductions\n"
        "• **AI Salary Prediction & Appraisal Increments**\n"
        "• **Payroll Anomaly & Audit Detection**\n\n"
        "Type any specific question or click one of the suggested prompts below!"
    )


# ---------------------------------------------------------
# Application Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    employee_count = Employee.query.filter_by(active=True).count()
    payroll_count = Payroll.query.count()
    latest_payrolls = Payroll.query.order_by(Payroll.created_at.desc()).limit(6).all()
    total_net = db.session.query(func.coalesce(func.sum(Payroll.net), 0.0)).scalar()
    total_gross = db.session.query(func.coalesce(func.sum(Payroll.gross), 0.0)).scalar()

    # Department breakdown for statistics
    dept_stats = (
        db.session.query(Employee.department, func.count(Employee.id), func.sum(Employee.monthly_basic))
        .filter(Employee.active == True)
        .group_by(Employee.department)
        .all()
    )

    # Anomaly summary
    all_payrolls = Payroll.query.all()
    anomaly_summary = detect_payroll_anomalies(all_payrolls)

    return render_template(
        "index.html",
        employee_count=employee_count,
        payroll_count=payroll_count,
        total_net=total_net,
        total_gross=total_gross,
        latest_payrolls=latest_payrolls,
        dept_stats=dept_stats,
        anomaly_summary=anomaly_summary,
    )


@app.route("/api/chart-data")
def chart_data():
    """Returns dynamic data for Chart.js dashboards."""
    # 1. Department basic salary distribution
    dept_query = (
        db.session.query(Employee.department, func.sum(Employee.monthly_basic))
        .filter(Employee.active == True)
        .group_by(Employee.department)
        .all()
    )
    dept_labels = [d[0] for d in dept_query]
    dept_values = [round(d[1], 2) for d in dept_query]

    # 2. Monthly payroll trend (Last 6 months)
    month_query = (
        db.session.query(Payroll.pay_month, func.sum(Payroll.net), func.sum(Payroll.gross))
        .group_by(Payroll.pay_month)
        .order_by(Payroll.pay_month.asc())
        .limit(6)
        .all()
    )
    month_labels = [m[0] for m in month_query]
    month_net = [round(m[1], 2) for m in month_query]
    month_gross = [round(m[2], 2) for m in month_query]

    return jsonify({
        "departments": {"labels": dept_labels, "data": dept_values},
        "monthly_trend": {"labels": month_labels, "net": month_net, "gross": month_gross},
    })


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Welcome back! Signed in successfully.", "success")
            return redirect(request.args.get("next") or url_for("index"))
        flash("Invalid username or password. Use demo login: admin / admin123", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been securely signed out.", "info")
    return redirect(url_for("index"))


# ---------------------------------------------------------
# Employee Management
# ---------------------------------------------------------

@app.route("/employees")
@admin_required
def employees():
    search = request.args.get("q", "").strip()
    department = request.args.get("dept", "").strip()

    query = Employee.query
    if search:
        query = query.filter(
            (Employee.name.ilike(f"%{search}%")) |
            (Employee.employee_code.ilike(f"%{search}%")) |
            (Employee.designation.ilike(f"%{search}%"))
        )
    if department:
        query = query.filter_by(department=department)

    records = query.order_by(Employee.name.asc()).all()
    all_depts = [d[0] for d in db.session.query(Employee.department).distinct().all()]

    return render_template(
        "employees.html",
        employees=records,
        search=search,
        selected_dept=department,
        departments=all_depts,
    )


@app.route("/employees/new", methods=["GET", "POST"])
@admin_required
def new_employee():
    if request.method == "POST":
        try:
            employee = Employee(
                employee_code=request.form["employee_code"].strip().upper(),
                name=request.form["name"].strip(),
                email=request.form["email"].strip(),
                phone=request.form.get("phone", "+91 9876543210").strip(),
                department=request.form["department"].strip(),
                designation=request.form["designation"].strip(),
                experience_years=int(request.form["experience_years"]),
                education=request.form["education"],
                performance_rating=int(request.form["performance_rating"]),
                monthly_basic=float(request.form["monthly_basic"]),
            )
            if employee.experience_years < 0 or employee.performance_rating not in range(1, 6) or employee.monthly_basic < 0:
                raise ValueError("Experience must be >= 0, rating between 1 and 5, and salary >= 0.")
            db.session.add(employee)
            db.session.commit()
            flash(f"Employee {employee.name} added successfully.", "success")
            return redirect(url_for("employees"))
        except (KeyError, ValueError) as exc:
            db.session.rollback()
            flash(f"Could not add employee: {exc}", "danger")
        except Exception:
            db.session.rollback()
            flash("Could not add employee. Make sure the Employee Code is unique.", "danger")
    return render_template("employee_form.html", employee=None)


@app.route("/employees/<int:employee_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_employee(employee_id):
    employee = db.get_or_404(Employee, employee_id)
    if request.method == "POST":
        try:
            employee.employee_code = request.form["employee_code"].strip().upper()
            employee.name = request.form["name"].strip()
            employee.email = request.form["email"].strip()
            employee.phone = request.form.get("phone", employee.phone).strip()
            employee.department = request.form["department"].strip()
            employee.designation = request.form["designation"].strip()
            employee.experience_years = int(request.form["experience_years"])
            employee.education = request.form["education"]
            employee.performance_rating = int(request.form["performance_rating"])
            employee.monthly_basic = float(request.form["monthly_basic"])
            if employee.experience_years < 0 or employee.performance_rating not in range(1, 6) or employee.monthly_basic < 0:
                raise ValueError("Experience must be >= 0, rating between 1 and 5, and salary >= 0.")
            db.session.commit()
            flash(f"Employee {employee.name} profile updated.", "success")
            return redirect(url_for("employees"))
        except Exception as exc:
            db.session.rollback()
            flash(f"Could not update employee: {exc}", "danger")
    return render_template("employee_form.html", employee=employee)


@app.route("/employees/<int:employee_id>/deactivate", methods=["POST"])
@admin_required
def deactivate_employee(employee_id):
    employee = db.get_or_404(Employee, employee_id)
    employee.active = False
    db.session.commit()
    flash(f"Employee {employee.name} deactivated.", "info")
    return redirect(url_for("employees"))


@app.route("/employees/<int:employee_id>/reactivate", methods=["POST"])
@admin_required
def reactivate_employee(employee_id):
    employee = db.get_or_404(Employee, employee_id)
    employee.active = True
    db.session.commit()
    flash(f"Employee {employee.name} reactivated.", "success")
    return redirect(url_for("employees"))


@app.route("/employees/export")
@admin_required
def export_employees_csv():
    records = Employee.query.order_by(Employee.employee_code.asc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Employee Code", "Name", "Email", "Phone", "Department", "Designation",
        "Experience (Yrs)", "Education", "Performance Rating", "Monthly Basic (INR)", "Status"
    ])
    for e in records:
        writer.writerow([
            e.employee_code, e.name, e.email, e.phone, e.department, e.designation,
            e.experience_years, e.education, e.performance_rating, e.monthly_basic,
            "Active" if e.active else "Inactive"
        ])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"employees_export_{date.today().strftime('%Y%m%d')}.csv"
    )


# ---------------------------------------------------------
# Attendance Management
# ---------------------------------------------------------

@app.route("/attendance")
@admin_required
def attendance():
    selected_month = request.args.get("month", date.today().strftime("%Y-%m"))
    employees = Employee.query.filter_by(active=True).order_by(Employee.name.asc()).all()

    # Load attendance for each employee
    attendance_records = {
        att.employee_id: att
        for att in Attendance.query.filter_by(month=selected_month).all()
    }

    return render_template(
        "attendance.html",
        employees=employees,
        attendance_records=attendance_records,
        selected_month=selected_month,
    )


@app.route("/attendance/log", methods=["POST"])
@admin_required
def log_attendance():
    try:
        employee_id = int(request.form["employee_id"])
        month = request.form["month"].strip()
        datetime.strptime(month, "%Y-%m")
        working_days = int(request.form.get("total_working_days", 22))
        present_days = int(request.form.get("present_days", 22))
        paid_leaves = int(request.form.get("paid_leaves", 0))
        unpaid_leaves = int(request.form.get("unpaid_leaves", 0))
        overtime_hours = float(request.form.get("overtime_hours", 0.0))

        if min(working_days, present_days, paid_leaves, unpaid_leaves, overtime_hours) < 0:
            raise ValueError("Attendance counts cannot be negative.")

        record = Attendance.query.filter_by(employee_id=employee_id, month=month).first()
        if not record:
            record = Attendance(employee_id=employee_id, month=month)
            db.session.add(record)

        record.total_working_days = working_days
        record.present_days = present_days
        record.paid_leaves = paid_leaves
        record.unpaid_leaves = unpaid_leaves
        record.overtime_hours = overtime_hours
        db.session.commit()

        flash("Attendance updated successfully.", "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Could not save attendance: {exc}", "danger")

    return redirect(url_for("attendance", month=request.form.get("month", date.today().strftime("%Y-%m"))))


@app.route("/attendance/auto-fill", methods=["POST"])
@admin_required
def auto_fill_attendance():
    month = request.form.get("month", date.today().strftime("%Y-%m"))
    active_employees = Employee.query.filter_by(active=True).all()
    count = 0
    for emp in active_employees:
        existing = Attendance.query.filter_by(employee_id=emp.id, month=month).first()
        if not existing:
            att = Attendance(
                employee_id=emp.id,
                month=month,
                total_working_days=22,
                present_days=22,
                paid_leaves=0,
                unpaid_leaves=0,
                overtime_hours=0.0
            )
            db.session.add(att)
            count += 1
    db.session.commit()
    flash(f"Auto-filled full attendance for {count} employees for {month}.", "success")
    return redirect(url_for("attendance", month=month))


# ---------------------------------------------------------
# Payroll Management
# ---------------------------------------------------------

@app.route("/payroll")
@admin_required
def payroll():
    month_filter = request.args.get("month", "")
    query = Payroll.query
    if month_filter:
        query = query.filter_by(pay_month=month_filter)

    records = query.order_by(Payroll.created_at.desc()).all()
    active_employees = Employee.query.filter_by(active=True).order_by(Employee.name.asc()).all()

    # Get distinct months in payroll
    available_months = [
        m[0] for m in db.session.query(Payroll.pay_month).distinct().order_by(Payroll.pay_month.desc()).all()
    ]

    return render_template(
        "payroll.html",
        payrolls=records,
        employees=active_employees,
        available_months=available_months,
        selected_month=month_filter,
    )


@app.route("/payroll/create", methods=["POST"])
@admin_required
def create_payroll():
    try:
        employee = db.get_or_404(Employee, int(request.form["employee_id"]))
        pay_month = request.form["pay_month"].strip()
        datetime.strptime(pay_month, "%Y-%m")

        allowance = float(request.form.get("allowance", 0.0))
        bonus = float(request.form.get("bonus", 0.0))
        deductions = float(request.form.get("deductions", 0.0))

        # Check if attendance LOP should be applied
        att = Attendance.query.filter_by(employee_id=employee.id, month=pay_month).first()
        if att and request.form.get("apply_attendance") == "1":
            lop = att.loss_of_pay_deduction
            ot = att.overtime_pay
            deductions += lop
            bonus += ot

        if min(allowance, bonus, deductions) < 0:
            raise ValueError("Allowances, bonuses, and deductions cannot be negative.")

        exists = Payroll.query.filter_by(employee_id=employee.id, pay_month=pay_month).first()
        if exists:
            raise ValueError(f"Payroll record already exists for {employee.name} in {pay_month}.")

        basic = employee.monthly_basic
        gross = round(basic + allowance + bonus, 2)
        net = round(gross - deductions, 2)
        if net < 0:
            raise ValueError(f"Deductions (₹{deductions:,.2f}) exceed gross pay (₹{gross:,.2f}).")

        record = Payroll(
            employee_id=employee.id,
            pay_month=pay_month,
            basic=basic,
            allowance=allowance,
            bonus=bonus,
            deductions=deductions,
            gross=gross,
            net=net,
            status="Paid",
            notes=f"Single run for {pay_month}"
        )
        db.session.add(record)
        db.session.commit()
        flash(f"Payroll for {employee.name} calculated and saved. Net: ₹{net:,.2f}", "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Payroll calculation failed: {exc}", "danger")

    return redirect(url_for("payroll"))


@app.route("/payroll/bulk-create", methods=["POST"])
@admin_required
def bulk_create_payroll():
    """One-click automated payroll calculation for all active employees for the selected month."""
    pay_month = request.form.get("pay_month", date.today().strftime("%Y-%m")).strip()
    try:
        datetime.strptime(pay_month, "%Y-%m")
    except ValueError:
        flash("Invalid pay month format.", "danger")
        return redirect(url_for("payroll"))

    active_employees = Employee.query.filter_by(active=True).all()
    created_count = 0
    skipped_count = 0

    for emp in active_employees:
        existing = Payroll.query.filter_by(employee_id=emp.id, pay_month=pay_month).first()
        if existing:
            skipped_count += 1
            continue

        basic = emp.monthly_basic
        # Standard HRA / Transport allowance = 15% of basic
        allowance = round(basic * 0.15, 2)
        # Standard statutory PF deduction = 12% of basic
        pf_deduction = round(basic * 0.12, 2)

        # Check attendance
        bonus = 0.0
        deductions = pf_deduction
        att = Attendance.query.filter_by(employee_id=emp.id, month=pay_month).first()
        if att:
            deductions += att.loss_of_pay_deduction
            bonus += att.overtime_pay

        gross = round(basic + allowance + bonus, 2)
        net = max(0.0, round(gross - deductions, 2))

        record = Payroll(
            employee_id=emp.id,
            pay_month=pay_month,
            basic=basic,
            allowance=allowance,
            bonus=bonus,
            deductions=deductions,
            gross=gross,
            net=net,
            status="Paid",
            notes=f"Automated bulk pay run for {pay_month}"
        )
        db.session.add(record)
        created_count += 1

    db.session.commit()
    flash(
        f"Bulk Pay Run Completed for {pay_month}: {created_count} payroll slips generated ({skipped_count} already existed).",
        "success"
    )
    return redirect(url_for("payroll", month=pay_month))


@app.route("/payroll/<int:payroll_id>")
@admin_required
def payslip(payroll_id):
    record = db.get_or_404(Payroll, payroll_id)
    att = Attendance.query.filter_by(employee_id=record.employee_id, month=record.pay_month).first()
    return render_template("payslip.html", record=record, attendance=att)


@app.route("/payroll/<int:payroll_id>/delete", methods=["POST"])
@admin_required
def delete_payroll(payroll_id):
    record = db.get_or_404(Payroll, payroll_id)
    db.session.delete(record)
    db.session.commit()
    flash("Payroll record deleted.", "info")
    return redirect(url_for("payroll"))


@app.route("/payroll/export")
@admin_required
def export_payroll_csv():
    records = Payroll.query.order_by(Payroll.pay_month.desc(), Payroll.created_at.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Payroll ID", "Month", "Employee Code", "Employee Name", "Department",
        "Basic (INR)", "Allowances (INR)", "Bonus (INR)", "Gross (INR)",
        "Deductions (INR)", "Net Pay (INR)", "Status", "Created At"
    ])
    for p in records:
        writer.writerow([
            p.id, p.pay_month, p.employee.employee_code, p.employee.name, p.employee.department,
            p.basic, p.allowance, p.bonus, p.gross, p.deductions, p.net, p.status,
            p.created_at.strftime("%Y-%m-%d %H:%M")
        ])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"payroll_report_{date.today().strftime('%Y%m%d')}.csv"
    )


@app.route("/payroll/anomalies")
@admin_required
def anomalies():
    """AI Anomaly & Discrepancy Detection Dashboard."""
    all_payrolls = Payroll.query.order_by(Payroll.pay_month.desc()).all()
    results = detect_payroll_anomalies(all_payrolls)
    return render_template("anomalies.html", audit=results)


# ---------------------------------------------------------
# Salary Prediction & Appraisal Engine
# ---------------------------------------------------------

@app.route("/salary-prediction", methods=["GET", "POST"])
@admin_required
def salary_prediction():
    result = None
    form_data = {
        "experience_years": "3",
        "education": "UG",
        "designation": "Python Developer",
        "performance_rating": "4"
    }
    if request.method == "POST":
        try:
            form_data = {
                "experience_years": request.form["experience_years"],
                "education": request.form["education"],
                "designation": request.form["designation"],
                "performance_rating": request.form["performance_rating"],
            }
            experience = int(form_data["experience_years"])
            rating = int(form_data["performance_rating"])
            if experience < 0 or experience > 50 or rating not in range(1, 6):
                raise ValueError("Experience must be 0–50 years and rating must be 1–5.")
            annual = predict_annual_salary(experience, form_data["education"], form_data["designation"], rating)
            result = {
                "annual": round(annual),
                "monthly": round(annual / 12),
                "min_range": round(annual * 0.90),
                "max_range": round(annual * 1.15),
            }
        except Exception as exc:
            flash(f"Prediction failed: {exc}", "danger")
    return render_template("prediction.html", result=result, form_data=form_data)


@app.route("/salary-appraisal", methods=["GET", "POST"])
@admin_required
def salary_appraisal():
    appraisal_result = None
    employees = Employee.query.filter_by(active=True).order_by(Employee.name.asc()).all()
    selected_emp_id = request.form.get("employee_id") or (employees[0].id if employees else None)

    if request.method == "POST" and selected_emp_id:
        emp = db.get_or_404(Employee, int(selected_emp_id))
        rating = int(request.form.get("rating", emp.performance_rating))
        market_salary = predict_annual_salary(emp.experience_years, emp.education, emp.designation, rating)
        appraisal_result = calculate_ai_appraisal(emp, emp.monthly_basic, rating, market_salary)
        appraisal_result["employee"] = emp

    return render_template(
        "appraisal.html",
        employees=employees,
        result=appraisal_result,
        selected_emp_id=int(selected_emp_id) if selected_emp_id else None,
    )


# ---------------------------------------------------------
# AI HR Chatbot
# ---------------------------------------------------------

@app.route("/chatbot", methods=["GET", "POST"])
@admin_required
def chatbot():
    answer = None
    question = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer = chatbot_reply(question)
        else:
            flash("Please enter a question.", "warning")
    return render_template(
        "chatbot.html",
        answer=answer,
        question=question,
        llm_enabled=bool(os.getenv("OPENAI_API_KEY", "").strip())
    )


# ---------------------------------------------------------
# Database Seeder (Demo Data)
# ---------------------------------------------------------

def seed_database():
    """Initializes default admin and realistic sample payroll & attendance data."""
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin123")

    # 1. Admin User
    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(username=username, role="admin")
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

    # 2. Sample Employees
    if Employee.query.count() == 0:
        samples = [
            Employee(
                employee_code="EMP001", name="Ananya Rao", email="ananya@paypilot.io",
                phone="+91 9845012345", department="Engineering", designation="Python Developer",
                experience_years=2, education="UG", performance_rating=4, monthly_basic=45000
            ),
            Employee(
                employee_code="EMP002", name="Rahul Kumar", email="rahul@paypilot.io",
                phone="+91 9845023456", department="Analytics", designation="Data Analyst",
                experience_years=3, education="PG", performance_rating=4, monthly_basic=52000
            ),
            Employee(
                employee_code="EMP003", name="Meera Sundaram", email="meera@paypilot.io",
                phone="+91 9845034567", department="Quality Assurance", designation="QA Engineer",
                experience_years=1, education="UG", performance_rating=3, monthly_basic=35000
            ),
            Employee(
                employee_code="EMP004", name="Vikram Adithya", email="vikram@paypilot.io",
                phone="+91 9845045678", department="Engineering", designation="Senior Developer",
                experience_years=6, education="PG", performance_rating=5, monthly_basic=85000
            ),
            Employee(
                employee_code="EMP005", name="Pooja Sharma", email="pooja@paypilot.io",
                phone="+91 9845056789", department="Human Resources", designation="HR Manager",
                experience_years=5, education="PG", performance_rating=4, monthly_basic=65000
            ),
            Employee(
                employee_code="EMP006", name="Karthik Raja", email="karthik@paypilot.io",
                phone="+91 9845067890", department="Finance", designation="Finance Manager",
                experience_years=5, education="PG", performance_rating=4, monthly_basic=70000
            ),
        ]
        db.session.add_all(samples)
        db.session.commit()

    # 3. Seed Attendance & Payroll for past & current months
    current_m = date.today().strftime("%Y-%m")
    all_emps = Employee.query.all()

    for emp in all_emps:
        # Check attendance
        att = Attendance.query.filter_by(employee_id=emp.id, month=current_m).first()
        if not att:
            # Add slight realistic variance
            unpaid = 1 if emp.employee_code == "EMP003" else 0
            ot = 4.0 if emp.employee_code == "EMP001" else 0.0
            att = Attendance(
                employee_id=emp.id,
                month=current_m,
                total_working_days=22,
                present_days=22 - unpaid,
                paid_leaves=0,
                unpaid_leaves=unpaid,
                overtime_hours=ot
            )
            db.session.add(att)

        # Check payroll
        pay = Payroll.query.filter_by(employee_id=emp.id, pay_month=current_m).first()
        if not pay:
            basic = emp.monthly_basic
            allowance = round(basic * 0.15, 2)
            bonus = 2000.0 if emp.performance_rating >= 4 else 0.0
            deductions = round(basic * 0.12, 2)  # 12% PF
            gross = round(basic + allowance + bonus, 2)
            net = round(gross - deductions, 2)
            pay = Payroll(
                employee_id=emp.id,
                pay_month=current_m,
                basic=basic,
                allowance=allowance,
                bonus=bonus,
                deductions=deductions,
                gross=gross,
                net=net,
                status="Paid",
                notes=f"Initial demo payroll for {current_m}"
            )
            db.session.add(pay)

    db.session.commit()


@app.cli.command("seed")
def seed():
    seed_database()
    print("Database successfully seeded with demo admin and records.")


# Ensure tables exist and default data is seeded automatically
with app.app_context():
    db.create_all()
    seed_database()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
