"""
工具函数模块
包含数据处理、模型训练和指标计算等辅助函数
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap


def load_file(uploaded_file):
    """加载上传的文件"""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None


def sanitize_numeric_df(df):
    """清理数据框，将字符串数字转为数值类型"""
    df2 = df.copy()
    numeric_cols_initial = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 先处理初始已经是数值类型的列
    for c in df2.columns:
        if df2[c].dtype == object:
            col = df2[c].astype(str).str.replace(',', '').str.replace('%','')
            # 尝试转换为数值类型
            try:
                converted_col = pd.to_numeric(col, errors='coerce')
                # 检查转换后非空值是否超过一定比例（比如90%）
                non_null_ratio = converted_col.notna().sum() / len(converted_col)
                if non_null_ratio > 0.5:  # 如果超过50%的值可以转换为数值
                    df2[c] = converted_col
            except Exception:
                pass
    
    # 最终获取所有数值类型的列
    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    return df2, numeric_cols


def train_model(df, features, target, n_estimators=200, random_state=42):
    """训练随机森林模型"""
    X = df[features].astype(float)
    y = df[target].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def metrics_for_model(model, X_train, X_test, y_train, y_test):
    """计算模型评估指标"""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=4, scoring='r2')
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
    except Exception:
        cv_mean = np.nan
        cv_std = np.nan
    return {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "cv_mean": cv_mean,
        "cv_std": cv_std
    }


def shap_values_from_model(model, X_background, X_explain):
    """计算SHAP值"""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_explain)
    return shap_vals, explainer


def safe_format(val, fmt=".3f"):
    """安全格式化数值"""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        v = float(val)
        return format(v, fmt)
    except Exception:
        return str(val)


def format_dataframe_numeric(df):
    """格式化数据框中的数值列"""
    fmt = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            fmt[c] = "{:.3f}"
    return fmt