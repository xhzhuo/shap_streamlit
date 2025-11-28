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


def train_model_with_tuning(df, features, target, 
                            n_iter=50, 
                            cv_folds=5,
                            random_state=42,
                            n_jobs=-1):
    """
    训练带参数调优的随机森林模型（使用RandomizedSearchCV）
    
    Parameters
    ----------
    df : DataFrame
        训练数据
    features : list
        特征列名列表
    target : str
        目标变量列名
    n_iter : int, default=50
        随机搜索的迭代次数（尝试的参数组合数）
    cv_folds : int, default=5
        交叉验证折数
    random_state : int, default=42
        随机种子
    n_jobs : int, default=-1
        并行作业数（-1表示使用所有CPU核心）
    
    Returns
    -------
    dict : {
        'best_model': 最优模型,
        'best_params': 最优参数字典,
        'best_score': 最优交叉验证分数,
        'cv_results': 完整的搜索结果,
        'X_train': 训练集特征,
        'X_test': 测试集特征,
        'y_train': 训练集目标,
        'y_test': 测试集目标,
        'default_score': 默认参数的分数（用于对比）
    }
    """
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    # 准备数据
    X = df[features].astype(float)
    y = df[target].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # 定义参数搜索空间
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'bootstrap': [True, False],
    }
    
    # 创建基础模型
    base_model = RandomForestRegressor(random_state=random_state, n_jobs=1)
    
    # 先训练一个默认参数的模型作为基准
    default_model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    default_model.fit(X_train, y_train)
    default_score = default_model.score(X_test, y_test)
    
    # 创建随机搜索对象
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='r2',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
        return_train_score=True
    )
    
    # 执行参数搜索
    random_search.fit(X_train, y_train)
    
    # 获取最优模型（使用所有CPU核心重新训练）
    best_params = random_search.best_params_
    best_model = RandomForestRegressor(**best_params, random_state=random_state, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'default_score': default_score,
        'search_object': random_search
    }


def metrics_for_model(model, X_train, X_test, y_train, y_test):
    """计算模型评估指标"""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # 计算归一化RMSE
    train_nrmse = normalized_rmse(y_train, y_train_pred, method='iqr')
    test_nrmse = normalized_rmse(y_test, y_test_pred, method='iqr')
    
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
        "train_nrmse": float(train_nrmse),
        "test_nrmse": float(test_nrmse),
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


def normalized_rmse(y_true, y_pred, method='iqr'):
    """
    计算归一化的RMSE (Normalized Root Mean Squared Error)
    
    Parameters:
    y_true: array-like, 真实值
    y_pred: array-like, 预测值
    method: str, 归一化方法
        - 'iqr': 使用四分位距 (Interquartile Range)
        - 'mean': 使用真实值的均值
        - 'range': 使用真实值的范围 (max - min)
        - 'std': 使用真实值的标准差
    
    Returns:
    float: 归一化的RMSE值
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if method == 'iqr':
        # 使用四分位距归一化
        q75, q25 = np.percentile(y_true, [75, 25])
        normalization_factor = q75 - q25
    elif method == 'mean':
        # 使用均值归一化
        normalization_factor = np.mean(y_true)
    elif method == 'range':
        # 使用范围归一化
        normalization_factor = np.max(y_true) - np.min(y_true)
    elif method == 'std':
        # 使用标准差归一化
        normalization_factor = np.std(y_true)
    else:
        raise ValueError("不支持的归一化方法")
    
    # 避免除以零
    if normalization_factor == 0:
        return np.inf if rmse > 0 else 0
    
    return rmse / normalization_factor
