from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR.parent / "Lab10_Final_Project.csv"
MODEL_PATH = APP_DIR / "best_logistic_regression.joblib"
NUMERIC_FEATURES = ["price", "delivery_days", "rating", "discount_percent"]
CATEGORICAL_FEATURES = ["category", "repeat_customer"]
TARGET = "returned"

st.set_page_config(
	page_title="Return Lens | วิเคราะห์การคืนสินค้า",
	page_icon="R",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
	:root {
		--ink: #152b28;
		--muted: #65736e;
		--paper: #f3f6f2;
		--surface: #ffffff;
		--line: #dce5dc;
		--green: #236b55;
		--green-soft: #e2efe7;
		--coral: #c2634a;
		--coral-soft: #f8e9e3;
	}
	.stApp { background: var(--paper); color: var(--ink); }
	.block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }
	h1, h2, h3, p, label, button, [data-testid="stMetricValue"],
	[data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {
		font-family: 'IBM Plex Sans Thai', sans-serif;
		color: var(--ink);
	}
	h1 { font-size: 2.25rem !important; font-weight: 700 !important; line-height: 1.25 !important; }
	h2, h3 { font-weight: 600 !important; }
	[data-testid="stSidebar"] { background: #e8eee8; border-right: 1px solid var(--line); }
	[data-testid="stSidebar"] * { font-family: 'IBM Plex Sans Thai', sans-serif; }
	[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
		font-family: 'Material Symbols Rounded' !important;
		font-weight: normal;
		font-style: normal;
		letter-spacing: normal;
		text-transform: none;
		white-space: nowrap;
		direction: ltr;
		-webkit-font-feature-settings: 'liga';
		-webkit-font-smoothing: antialiased;
	}
	.eyebrow {
		color: var(--green); font: 800 0.72rem 'Manrope', sans-serif;
		letter-spacing: 1.3px; text-transform: uppercase; margin-bottom: 0.35rem;
	}
	.page-intro { color: var(--muted); font-size: 1rem; margin-top: -0.45rem; }
	.section-kicker { color: var(--muted); font-size: 0.8rem; margin-bottom: 0.15rem; }
	div[data-testid="stMetric"] {
		background: var(--surface); border: 1px solid var(--line); border-radius: 7px;
		padding: 0.9rem 1rem; min-height: 112px;
	}
	div[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.82rem; }
	div[data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; font-weight: 800; }
	div[data-testid="stForm"] {
		background: var(--surface); border: 1px solid var(--line); border-radius: 7px;
		padding: 1.25rem 1.35rem;
	}
	div.stButton > button[kind="primary"],
	div.stFormSubmitButton > button[kind="primary"] {
		background: var(--green); border: 1px solid var(--green); color: white;
		border-radius: 5px; min-height: 2.8rem; font-weight: 700;
	}
	div.stButton > button[kind="primary"]:hover,
	div.stFormSubmitButton > button[kind="primary"]:hover { background: #194f40; color: white; }
	div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
		border-radius: 5px; border-color: var(--line);
	}
	[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }
	hr { border-color: var(--line); }
	</style>
	""",
	unsafe_allow_html=True,
)


def clean_overview_data(data: pd.DataFrame) -> pd.DataFrame:
	cleaned = data.copy()
	if "order_id" in cleaned.columns:
		cleaned["order_id"] = cleaned["order_id"].astype("string").str.strip()
		cleaned = cleaned.drop_duplicates(subset="order_id", keep="first")

	for column in NUMERIC_FEATURES:
		cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
	cleaned.loc[~cleaned["rating"].between(1, 5), "rating"] = np.nan
	cleaned[TARGET] = pd.to_numeric(cleaned[TARGET], errors="coerce")
	cleaned = cleaned[cleaned[TARGET].isin([0, 1])].copy()

	for column in CATEGORICAL_FEATURES:
		cleaned[column] = cleaned[column].astype("string").str.strip()
		cleaned[column] = cleaned[column].replace("", pd.NA).str.casefold().str.title()

	for column in NUMERIC_FEATURES:
		median = cleaned[column].median()
		cleaned[column] = cleaned[column].fillna(0 if pd.isna(median) else median)
	for column in CATEGORICAL_FEATURES:
		mode = cleaned[column].mode(dropna=True)
		cleaned[column] = cleaned[column].fillna(mode.iloc[0] if not mode.empty else "ไม่ระบุ")

	cleaned[TARGET] = cleaned[TARGET].astype(int)
	return cleaned


@st.cache_resource(show_spinner="กำลังเตรียมโมเดลและข้อมูล...")
def load_assets(
	data_path: str,
	model_path: str,
	data_modified: int,
	model_modified: int,
) -> dict:
	data = pd.read_csv(data_path)
	required_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET]
	missing_columns = [column for column in required_columns if column not in data.columns]
	if missing_columns:
		raise ValueError(f"CSV ไม่มีคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")

	features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
	target = pd.to_numeric(data[TARGET], errors="raise").astype(int)
	train_indices, test_indices = train_test_split(
		np.arange(len(data)),
		test_size=0.2,
		random_state=42,
		stratify=target,
	)

	numeric_pipeline = Pipeline([
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	])
	categorical_pipeline = Pipeline([
		("imputer", SimpleImputer(strategy="most_frequent")),
		("encoder", OneHotEncoder(handle_unknown="ignore")),
	])
	preprocessor = ColumnTransformer([
		("numeric", numeric_pipeline, NUMERIC_FEATURES),
		("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
	])

	train_features = features.iloc[train_indices]
	train_target = target.iloc[train_indices]
	train_transformed = preprocessor.fit_transform(train_features)
	feature_names = preprocessor.get_feature_names_out()
	feature_selector = SelectKBest(
		score_func=lambda values, labels: mutual_info_classif(
			values, labels, random_state=42
		),
		k=min(8, train_transformed.shape[1]),
	)
	train_selected = feature_selector.fit_transform(train_transformed, train_target)

	model = joblib.load(model_path)
	if model.n_features_in_ != train_selected.shape[1]:
		raise ValueError(
			f"โมเดลรับ {model.n_features_in_} features แต่ข้อมูลที่เตรียมได้ "
			f"{train_selected.shape[1]} features"
		)

	test_features = features.iloc[test_indices]
	test_target = target.iloc[test_indices]
	test_selected = feature_selector.transform(preprocessor.transform(test_features))
	test_predictions = model.predict(test_selected)
	test_probabilities = model.predict_proba(test_selected)[:, list(model.classes_).index(1)]
	selected_names = feature_names[feature_selector.get_support()]
	return {
		"data": data,
		"clean_data": clean_overview_data(data),
		"features": features,
		"target": target,
		"preprocessor": preprocessor,
		"selector": feature_selector,
		"selected_names": selected_names,
		"model": model,
		"test_target": test_target,
		"test_predictions": test_predictions,
		"test_probabilities": test_probabilities,
		"metrics": {
			"accuracy": accuracy_score(test_target, test_predictions),
			"precision": precision_score(test_target, test_predictions, zero_division=0),
			"recall": recall_score(test_target, test_predictions, zero_division=0),
			"f1": f1_score(test_target, test_predictions, zero_division=0),
		},
	}


if not DATA_PATH.is_file():
	st.error(f"ไม่พบชุดข้อมูล: {DATA_PATH}")
	st.stop()
if not MODEL_PATH.is_file():
	st.error(f"ไม่พบไฟล์โมเดล: {MODEL_PATH}")
	st.stop()

try:
	assets = load_assets(
		str(DATA_PATH),
		str(MODEL_PATH),
		DATA_PATH.stat().st_mtime_ns,
		MODEL_PATH.stat().st_mtime_ns,
	)
except Exception as error:
	st.error(f"เตรียมแอปไม่สำเร็จ: {error}")
	st.stop()

data = assets["data"]
clean_data = assets["clean_data"]
selected_names = assets["selected_names"]
selected_categories = any(name.startswith("categorical__category_") for name in selected_names)
selected_repeat_customer = any(
	name.startswith("categorical__repeat_customer_") for name in selected_names
)

st.markdown('<div class="eyebrow">RETURNS ANALYTICS / 01</div>', unsafe_allow_html=True)
st.title("Return Lens")
st.markdown(
	'<p class="page-intro">มองเห็นรูปแบบการคืนสินค้า และประเมินความเสี่ยงของคำสั่งซื้อด้วย Logistic Regression</p>',
	unsafe_allow_html=True,
)

with st.sidebar:
	st.markdown('<div class="eyebrow">WORKSPACE</div>', unsafe_allow_html=True)
	st.markdown("### Return Lens")
	st.caption("Order intelligence")
	st.divider()
	threshold = st.slider(
		"เกณฑ์จัดกลุ่มความเสี่ยง",
		min_value=0.10,
		max_value=0.90,
		value=0.50,
		step=0.05,
		format="%.2f",
		help="ปรับเกณฑ์ตัดสินจากความน่าจะเป็นสำหรับหน้าประเมินคำสั่งซื้อ",
	)
	st.caption(f"ใช้ข้อมูล clean แล้ว {len(clean_data):,} ออเดอร์")
	st.caption("Model: best_logistic_regression")

overview_tab, prediction_tab, data_tab = st.tabs(
	["ภาพรวม", "ประเมินคำสั่งซื้อ", "สำรวจข้อมูล"]
)

with overview_tab:
	st.markdown('<div class="section-kicker">DATA SNAPSHOT</div>', unsafe_allow_html=True)
	st.subheader("ภาพรวมคำสั่งซื้อ")
	total_orders = len(clean_data)
	return_rate = float(clean_data[TARGET].mean())
	average_price = float(clean_data["price"].mean())
	test_metrics = assets["metrics"]
	metric_columns = st.columns(4)
	metric_columns[0].metric("คำสั่งซื้อทั้งหมด", f"{total_orders:,}")
	metric_columns[1].metric("อัตราคืนสินค้า", f"{return_rate:.1%}")
	metric_columns[2].metric("ราคาเฉลี่ย", f"฿{average_price:,.0f}")
	metric_columns[3].metric("F1 บนชุดทดสอบ", f"{test_metrics['f1']:.1%}")

	st.divider()
	chart_left, chart_right = st.columns([1.15, 0.85], gap="large")
	with chart_left:
		st.markdown('<div class="section-kicker">RETURN RATE BY CATEGORY</div>', unsafe_allow_html=True)
		st.subheader("อัตราคืนแยกตามประเภทสินค้า")
		category_summary = (
			clean_data.assign(returned_numeric=clean_data[TARGET])
			.groupby("category", dropna=False)["returned_numeric"]
			.mean()
			.sort_values(ascending=False)
		)
		category_summary.index = category_summary.index.fillna("ไม่ระบุ")
		st.bar_chart(category_summary, color="#236b55", height=270)
	with chart_right:
		st.markdown('<div class="section-kicker">MODEL CHECK / HOLDOUT</div>', unsafe_allow_html=True)
		st.subheader("ผลทดสอบโมเดล")
		holdout_metrics = pd.Series(
			{
				"Accuracy": test_metrics["accuracy"],
				"Precision": test_metrics["precision"],
				"Recall": test_metrics["recall"],
				"F1-score": test_metrics["f1"],
			}
		)
		st.bar_chart(holdout_metrics, color="#c2634a", height=270)
		st.caption("ชุดทดสอบ 20% · stratified split · random state 42")

	st.divider()
	st.markdown('<div class="section-kicker">CLEANED DATA</div>', unsafe_allow_html=True)
	st.subheader("คำสั่งซื้อหลังทำความสะอาด")
	st.caption(
		f"เหลือ {len(clean_data):,} รายการจาก {len(data):,} รายการ: ตัด order ID ซ้ำ, "
		"รวมหมวดหมู่ที่ต่างกันเฉพาะตัวพิมพ์, เติมค่าว่าง และจัดการคะแนนรีวิวให้อยู่ในช่วง 1–5"
	)
	st.dataframe(clean_data, width="stretch", hide_index=True, height=360)
	st.download_button(
		"ดาวน์โหลดข้อมูลที่ clean แล้ว",
		data=clean_data.to_csv(index=False).encode("utf-8-sig"),
		file_name="orders_cleaned.csv",
		mime="text/csv",
	)

	st.divider()
	st.caption(
		"การประเมินนี้ใช้ชุดทดสอบชุดเดียว และผลจากโมเดลเป็นความน่าจะเป็นเชิงสถิติ "
		"ไม่ใช่ข้อสรุปเชิงเหตุและผล"
	)

with prediction_tab:
	st.markdown('<div class="section-kicker">QUICK ASSESSMENT</div>', unsafe_allow_html=True)
	st.subheader("ประเมินความเสี่ยงรายออเดอร์")
	st.caption("กรอกข้อมูลสินค้าและการจัดส่ง ระบบจะคำนวณโอกาสคืนจากโมเดลที่ฝึกไว้")

	with st.form("return_risk_form"):
		left, right = st.columns(2, gap="large")
		with left:
			price = st.number_input(
				"ราคาสินค้า (บาท)", min_value=0.0, value=float(data["price"].median()), step=50.0
			)
			delivery_days = st.number_input(
				"ระยะเวลาจัดส่ง (วัน)",
				min_value=0,
				value=int(data["delivery_days"].median()),
				step=1,
			)
			rating = st.select_slider("คะแนนรีวิว", options=[1, 2, 3, 4, 5], value=3)
		with right:
			discount_percent = st.number_input(
				"ส่วนลด (%)",
				min_value=0.0,
				max_value=100.0,
				value=float(data["discount_percent"].median()),
				step=1.0,
			)
			if selected_repeat_customer:
				customer_values = sorted(data["repeat_customer"].dropna().astype(str).unique())
				repeat_customer = st.selectbox("ลูกค้าซื้อซ้ำ", customer_values)
			else:
				repeat_customer = data["repeat_customer"].mode().iloc[0]
			if selected_categories:
				category_values = sorted(data["category"].dropna().astype(str).unique())
				category = st.selectbox("ประเภทสินค้า", category_values)
			else:
				category = data["category"].mode().iloc[0]
				st.info("ฟีเจอร์ประเภทสินค้าไม่ได้อยู่ใน 8 ฟีเจอร์ที่โมเดลเลือก จึงไม่นำมาใช้คำนวณ")

		submitted = st.form_submit_button(
			"ประเมินความเสี่ยง", type="primary", width="stretch"
		)

	if submitted:
		order = pd.DataFrame([{
			"price": price,
			"delivery_days": delivery_days,
			"rating": rating,
			"discount_percent": discount_percent,
			"category": category,
			"repeat_customer": repeat_customer,
		}])[assets["features"].columns]
		prepared_order = assets["preprocessor"].transform(order)
		selected_order = assets["selector"].transform(prepared_order)
		model = assets["model"]
		return_probability = float(
			model.predict_proba(selected_order)[0, list(model.classes_).index(1)]
		)
		is_high_risk = return_probability >= threshold

		st.divider()
		result_column, detail_column = st.columns([1.2, 0.8], gap="large")
		with result_column:
			if is_high_risk:
				st.warning("มีแนวโน้มคืนสินค้า")
			else:
				st.success("มีแนวโน้มไม่คืนสินค้า")
			st.metric("โอกาสคืนสินค้า", f"{return_probability:.1%}")
			st.progress(return_probability, text=f"เกณฑ์ปัจจุบัน {threshold:.0%}")
		with detail_column:
			st.markdown("#### ข้อมูลที่ประเมิน")
			st.write(f"ราคา: ฿{price:,.2f}")
			st.write(f"ระยะเวลาจัดส่ง: {delivery_days} วัน")
			st.write(f"คะแนนรีวิว: {rating} / 5")
			st.write(f"ส่วนลด: {discount_percent:.0f}%")
			if selected_repeat_customer:
				st.write(f"ลูกค้าซื้อซ้ำ: {repeat_customer}")
		st.caption("ผลทำนายขึ้นกับข้อมูลและ preprocessing ที่ใช้ฝึกโมเดล")

with data_tab:
	st.markdown('<div class="section-kicker">DATA EXPLORER</div>', unsafe_allow_html=True)
	st.subheader("สำรวจชุดข้อมูล")
	filter_columns = st.columns([1.2, 1, 1])
	search_text = filter_columns[0].text_input("ค้นหาข้อมูล")
	category_values = sorted(data["category"].dropna().astype(str).unique())
	category_filter = filter_columns[1].selectbox("ประเภทสินค้า", ["ทั้งหมด", *category_values])
	return_filter = filter_columns[2].selectbox(
		"สถานะคืนสินค้า", ["ทั้งหมด", "คืนสินค้า", "ไม่คืนสินค้า"]
	)

	filtered_data = data.copy()
	if category_filter != "ทั้งหมด":
		filtered_data = filtered_data[filtered_data["category"].astype(str) == category_filter]
	if return_filter != "ทั้งหมด":
		target_value = 1 if return_filter == "คืนสินค้า" else 0
		filtered_data = filtered_data[pd.to_numeric(filtered_data[TARGET], errors="coerce") == target_value]
	if search_text:
		matches = filtered_data.astype(str).apply(
			lambda column: column.str.contains(search_text, case=False, regex=False, na=False)
		).any(axis=1)
		filtered_data = filtered_data[matches]

	data_metrics = st.columns(3)
	data_metrics[0].metric("จำนวนข้อมูล", f"{len(data):,}")
	data_metrics[1].metric("หลังกรอง", f"{len(filtered_data):,}")
	if len(filtered_data):
		filtered_return_rate = pd.to_numeric(filtered_data[TARGET], errors="coerce").mean()
		data_metrics[2].metric("อัตราคืนในผลลัพธ์", f"{filtered_return_rate:.1%}")

	row_limit = st.select_slider(
		"จำนวนแถวที่แสดง",
		options=sorted(set([10, 25, 50, 100, max(1, len(filtered_data))])),
		value=min(25, max(1, len(filtered_data))),
	)
	st.dataframe(filtered_data.head(row_limit), width="stretch", hide_index=True)
	st.download_button(
		"ดาวน์โหลดข้อมูลที่กรอง",
		data=filtered_data.to_csv(index=False).encode("utf-8-sig"),
		file_name="orders_filtered.csv",
		mime="text/csv",
	)

st.divider()
st.caption("Model artifact: best_logistic_regression.joblib · Source: Lab10_Final_Project.csv")
