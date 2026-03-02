# app.py - Streamlit血压分类预测应用
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
import sys
import os

# 设置页面配置
st.set_page_config(
    page_title="血压分类预测工具",
    page_icon="❤️",
    layout="wide"
)

# 应用标题和说明
st.title("❤️ 血压分类预测工具")
st.markdown("### 基于机器学习模型的血压异常检测系统")
st.markdown("填写生理参数，预测血压是否正常")


# 加载配置和模型
@st.cache_resource
def load_model_and_config():
    """加载模型和配置"""
    try:
        # 加载Streamlit配置
        with open('streamlit_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 加载模型
        with open(config['model_file'], 'rb') as f:
            model = pickle.load(f)

        # 加载标准化器
        with open(config['scaler_file'], 'rb') as f:
            scaler = pickle.load(f)

        # 加载特征名称
        feature_names = config['feature_names']
        optimal_threshold = config['optimal_threshold']

        return model, scaler, feature_names, optimal_threshold, config

    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        return None, None, None, None, None


# 加载模型
model, scaler, feature_names, optimal_threshold, config = load_model_and_config()

if model is None:
    st.warning("⚠️ 无法加载模型，请确保模型文件存在。")
    st.stop()

# 显示模型信息
with st.sidebar:
    st.header("📊 模型信息")
    st.markdown(f"**模型类型**: {config.get('model_name', 'XGBoost')}")
    st.markdown(f"**特征数量**: {len(feature_names)}")
    st.markdown(f"**最优阈值**: {optimal_threshold:.3f}")

    # 显示性能指标（如果存在）
    if 'performance_metrics' in config:
        st.markdown("---")
        st.subheader("模型性能")
        metrics = config['performance_metrics']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC", f"{metrics.get('AUC', 0):.3f}")
            st.metric("准确率", f"{metrics.get('Accuracy', 0):.3f}")
        with col2:
            st.metric("F1分数", f"{metrics.get('F1_score', 0):.3f}")
            st.metric("召回率", f"{metrics.get('Recall', 0):.3f}")

    st.markdown("---")
    st.info("💡 **使用说明**: 填写所有参数后点击预测按钮")

# 主内容区 - 输入表单
st.header("📝 输入参数")

# 创建输入列
col1, col2 = st.columns(2)

with col1:
    st.subheader("基本生理参数")
    age = st.slider("**年龄**", 18, 90, 50, help="选择年龄（18-90岁）")
    weight = st.slider("**体重 (kg)**", 40.0, 120.0, 60.0, 0.1, help="输入体重（40-120kg）")
    height = st.slider("**身高 (cm)**", 140.0, 200.0, 165.0, 0.1, help="输入身高（140-200cm）")
    bmi = st.slider("**BMI**", 15.0, 40.0, 22.0, 0.1, help="身体质量指数（15-40）")

with col2:
    st.subheader("医疗参数")
    alt = st.slider("**ALT (U/L)**", 0.0, 100.0, 20.0, 0.1, help="丙氨酸氨基转移酶（0-100 U/L）")
    pulse = st.slider("**脉搏 (次/分)**", 40, 120, 72, help="脉搏次数（40-120次/分）")

    st.subheader("时间参数")
    month_sin = st.slider("**月份正弦值**", -1.0, 1.0, 0.0, 0.01, help="月份的正弦值（-1到1）")
    month_cos = st.slider("**月份余弦值**", -1.0, 1.0, 1.0, 0.01, help="月份的余弦值（-1到1）")
    hour_sin = st.slider("**小时正弦值**", -1.0, 1.0, 0.0, 0.01, help="小时的正弦值（-1到1）")
    hour_cos = st.slider("**小时余弦值**", -1.0, 1.0, 1.0, 0.01, help="小时的余弦值（-1到1）")

# 预测按钮和结果区域
st.markdown("---")
st.header("🔮 预测结果")

# 创建预测按钮列
predict_col, result_col = st.columns([1, 2])

with predict_col:
    predict_button = st.button("开始预测", type="primary", use_container_width=True)

# 初始化结果状态
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
    st.session_state.prediction_prob = None
    st.session_state.feature_values = None

# 处理预测
if predict_button:
    with st.spinner("正在计算..."):
        try:
            # 准备特征数据（必须按照训练时的特征顺序）
            feature_values = {
                'age': age,
                'alt': alt,
                'height': height,
                'weight': weight,
                'pulse': pulse,
                'BMI': bmi,
                'month_sin': month_sin,
                'month_cos': month_cos,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos
            }

            # 确保所有特征都存在
            input_features = []
            for feature in feature_names:
                if feature in feature_values:
                    input_features.append(feature_values[feature])
                else:
                    # 如果特征不存在，使用默认值
                    st.warning(f"特征 '{feature}' 不在输入中，使用默认值0")
                    input_features.append(0)

            # 转换为numpy数组
            X_input = np.array([input_features])

            # 数据标准化
            X_scaled = scaler.transform(X_input)

            # 预测
            probability = model.predict_proba(X_scaled)[0, 1]
            prediction = 1 if probability >= optimal_threshold else 0

            # 保存结果到session state
            st.session_state.prediction_result = prediction
            st.session_state.prediction_prob = probability
            st.session_state.feature_values = feature_values

        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")

# 显示结果
with result_col:
    if st.session_state.prediction_result is not None:
        probability = st.session_state.prediction_prob
        prediction = st.session_state.prediction_result

        # 显示概率仪表
        st.subheader("预测结果")

        # 创建两列显示结果
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            # 血压状态指示
            if prediction == 0:
                st.success("✅ **正常血压**")
                st.metric("异常概率", f"{probability:.3f}")
            else:
                st.error("⚠️ **异常血压**")
                st.metric("异常概率", f"{probability:.3f}")

            # 阈值指示
            st.progress(float(probability))
            st.caption(f"阈值: {optimal_threshold:.3f}")

        with result_col2:
            # 显示详细信息
            st.markdown("#### 详细分析")

            # 计算风险等级
            if probability < 0.3:
                risk_level = "低风险"
                risk_color = "green"
            elif probability < 0.7:
                risk_level = "中风险"
                risk_color = "orange"
            else:
                risk_level = "高风险"
                risk_color = "red"

            st.markdown(f"**风险等级**: :{risk_color}[{risk_level}]")

            # 建议
            if prediction == 0:
                st.markdown("**建议**: 保持健康生活方式，定期监测血压")
            else:
                st.markdown("**建议**: 建议进行进一步检查并咨询医生")

        # 显示输入参数回顾
        st.markdown("---")
        with st.expander("📋 查看输入参数"):
            if st.session_state.feature_values:
                param_df = pd.DataFrame(
                    list(st.session_state.feature_values.items()),
                    columns=['参数', '值']
                )
                st.table(param_df)

    else:
        st.info("👆 点击'开始预测'按钮查看结果")

# 特征重要性展示（如果可用）
try:
    if hasattr(model, 'feature_importances_'):
        st.markdown("---")
        st.header("📊 特征重要性分析")

        # 获取特征重要性
        importances = model.feature_importances_

        # 创建DataFrame
        importance_df = pd.DataFrame({
            '特征': feature_names,
            '重要性': importances
        }).sort_values('重要性', ascending=False)

        # 显示条形图
        st.bar_chart(importance_df.set_index('特征')['重要性'])

        # 显示表格
        with st.expander("查看详细特征重要性"):
            st.dataframe(importance_df, use_container_width=True)

except Exception as e:
    # 如果无法获取特征重要性，跳过
    pass

# 血压分类标准参考
st.markdown("---")
st.header("📚 血压分类标准参考")

col_ref1, col_ref2 = st.columns(2)

with col_ref1:
    st.markdown("""
    ### 血压分类标准
    | 分类 | 收缩压 (mmHg) | 舒张压 (mmHg) |
    |------|---------------|---------------|
    | 正常血压 | < 120 | < 80 |
    | 正常高值 | 120-129 | 80-84 |
    | 1级高血压 | 130-139 | 85-89 |
    | 2级高血压 | 140-159 | 90-99 |
    | 3级高血压 | ≥ 160 | ≥ 100 |
    """)

with col_ref2:
    st.markdown("""
    ### 注意事项
    1. **本工具为预测工具**，结果仅供参考
    2. **实际诊断**请以医疗机构测量为准
    3. **建议**定期监测血压
    4. **保持**健康生活方式
    5. **如有异常**及时就医
    """)

# 页脚
st.markdown("---")
st.caption("💡 注意：本工具基于机器学习模型预测，准确率受训练数据影响")
st.caption("🔒 隐私保护：所有计算在服务器完成，不保存任何用户数据")
st.caption(f"🔄 最后更新: {config.get('training_info', {}).get('deployment_date', '未知')}")

# 调试信息（仅在开发模式下显示）
if st.sidebar.checkbox("显示调试信息", False):
    st.sidebar.markdown("---")
    st.sidebar.subheader("调试信息")
    st.sidebar.json(config)
    st.sidebar.write(f"特征名称: {feature_names}")
    st.sidebar.write(f"特征数量: {len(feature_names)}")
