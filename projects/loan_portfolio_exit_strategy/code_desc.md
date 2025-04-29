# 贷款组合退出策略模拟 (`simulate_two_strategies.py`) 代码逻辑与假设总结 (中文)

本文档旨在概述 `simulate_two_strategies.py` 脚本中使用的核心假设和模拟逻辑，以便于代码审查。

## 1. 全局参数与假设

### 1.1 投资组合基础设定
*   **总贷款数 (`N_LOANS`)**: 10,000 笔
*   **平均贷款余额 (`AVG_BALANCE`)**: 100,000 美元
*   **风险分箱数 (`N_BINS`)**: 8 个 (索引 0-7)
*   **每箱贷款数 (`LOANS_PER_BIN`)**: 1,250 笔 (假设初始均匀分布)
*   **总组合价值 (`TOTAL_PORTFOLIO_VALUE`)**: 10亿 美元 (初始)

### 1.2 风险与利率
*   **年化利率 (`INTEREST_RATE_ANNUAL`)**: 12% (APR)
*   **月利率 (`INTEREST_RATE_MONTHLY`)**: 1%
*   **各分箱年化违约率 (`PD_RATES`)**: 一个包含8个值的数组，风险随分箱指数递增 (Bin 0: 2%, Bin 7: 35%)
*   **高风险分箱 (`HIGH_RISK_BINS`)**: 定义为索引 6 和 7 (即 Bin 7 和 Bin 8)
*   **标准回收率 (`STANDARD_RECOVERY_RATE`)**: 30% (用于基线、S1低风险、S2/S3扩展期间违约、S3拒绝者)

### 1.3 资金与折现
*   **年化资金成本 (`FUNDING_COST_RATE_ANNUAL`)**: 3%
*   **净现值(NPV)折现率 (`DISCOUNT_RATE`)**: 使用年化资金成本率 (3%)

### 1.4 多年模拟动态
*   **客户流失率 (`CHURN_RATE`)**: 每年 30% 的活跃贷款会流失（提前还清），在每年年初应用。
*   **模拟年限 (`SIMULATION_YEARS`)**: 默认为 3 年 (注意：策略2强制为2年)。

### 1.5 策略特定参数
*   **策略1 - 回收率 (`RECOVERY_RATES_STRAT1`)**: 一个数组，定义了策略1中对高风险贷款强制回收时的不同回收率场景。
*   **策略2 & 3 - 违约降低系数 (`DEFAULT_REDUCTION_FACTORS`)**: 一个数组，定义了策略2/3中，针对高风险接受者，其年化PD降低的比例。
*   **策略2 & 3 - 延期月数 (`EXTENSION_MONTHS`)**: 24个月 (策略2使用) 或 12个月 (策略3使用)。
*   **策略2 & 3 - 本金减免系数 (`PRINCIPAL_REDUCTION_FACTOR`)**: 10% (针对高风险接受者，本金直接减少10%)。
*   **策略2 & 3 - 拒绝概率 (`REFUSAL_PROBABILITY`)**: 5% (高风险客户拒绝延期方案的概率)。
*   **策略2 & 3 - 延期利率增加 (`EXTENSION_RATE_INCREASE`)**: 5% (对延期贷款收取的年化利率**绝对值**增加 0.05)。
*   **策略2 - 风险率比率 (`HAZARD_RATE_RATIO`)**: 3.0 (策略2分段生存模型中，前6个月的风险率是后18个月的3倍)。

## 2. 模拟逻辑

### 2.1 基线策略 (`simulate_baseline_multiyear`)
*   **贷款类型**: 标准12个月气球贷（每月付息，到期还本）。
*   **模拟周期**: `SIMULATION_YEARS` 年。
*   **每年逻辑**:
    1.  年初应用 `CHURN_RATE` 减少各分箱贷款数。
    2.  计算年初组合价值，用于计算当年资金成本。
    3.  对剩余贷款，根据其所在分箱的 `PD_RATES` 计算预期违约数和生存数。
    4.  **损益计算**:
        *   利润 = 生存贷款支付的11个月利息。
        *   损失 = 违约贷款的本金损失 (本金 * (1 - `STANDARD_RECOVERY_RATE`))。
    5.  计算当年毛利润 (利润 - 本金损失) 和总绝对损失 (本金损失)。
    6.  计算当年资金成本 (年初价值 * `FUNDING_COST_RATE_ANNUAL`)。
    7.  计算当年税后净利润 (毛利润 - 资金成本)。
    8.  更新状态：只有生存的贷款进入下一年，余额不变。
*   **最终指标**: 计算整个模拟期间的年度税后净利润的NPV和总绝对损失。

### 2.2 策略1: 尾部断贷 (`simulate_strategy1_multiyear`)
*   **与基线区别**: 仅在于高风险分箱的损失计算。
*   **低风险分箱**: 逻辑同基线。
*   **高风险分箱**:
    *   利润计算同基线（生存者付息）。
    *   损失计算使用**特定的** `recovery_rate_recall` 参数 (本金 * (1 - `recovery_rate_recall`))。
*   **其他**: 流程（流失、资金成本、NPV计算）与基线相同。

### 2.3 策略2: 强制24个月分期展期 (`simulate_strategy2_multiyear`)
*   **重要**: 此策略强制模拟 **2年**。
*   **核心逻辑**: 将第一年开始时（流失后）的所有贷款转换为24个月分期摊还贷款，并计算这两年的总损益。
*   **年 1**:
    1.  年初应用 `CHURN_RATE`。记录进入展期的初始组合价值 (`value_entering_yr1_ext`)。
    2.  **分箱处理**:
        *   **低风险**:
            *   进入24个月分期摊还，利率为 `INTEREST_RATE_ANNUAL`。
            *   使用 `_calculate_s2_low_risk_outcome` 辅助函数，基于原始 `PD_RATES` 和分段生存模型 (`calculate_piecewise_survival_probs`) 计算24个月的总损益和总损失。
            *   将计算出的总损益/损失 **平均分配** 到第1年和第2年。
            *   记录这些贷款价值 (`value_entering_yr2_ext`) 用于第2年资金成本计算。
        *   **高风险**:
            *   **拒绝者** (`REFUSAL_PROBABILITY`): 按基线逻辑处理（付息或违约），损益和损失**全部计入第1年**。
            *   **接受者**:
                *   立即承受本金减免损失 (`initial_haircut_loss`)，计入**第1年损失**。
                *   进入24个月分期摊还，本金为减免后 (`reduced_principal`)，利率为 `INTEREST_RATE_ANNUAL + EXTENSION_RATE_INCREASE`。
                *   使用 `_calculate_s2_high_risk_acceptor_outcome` 辅助函数，基于**降低后**的年化PD (`pd_rate_annual * (1 - reduction_factor)`) 和分段生存模型计算24个月的总损益/损失（此函数内部已包含减免损失的影响）。
                *   将扣除减免损失后的总损益/损失 **平均分配** 到第1年和第2年。
                *   记录这些贷款的减免后价值 (`value_entering_yr2_ext`) 用于第2年资金成本计算。
    3.  **年 1 计算**: 汇总所有分箱分配到第1年的损益和损失。计算第1年资金成本 (基于 `value_entering_yr1_ext`)。计算第1年净损益。
*   **年 2**:
    1.  **计算**: 汇总所有分箱分配到第2年的损益和损失。计算第2年资金成本 (基于 `value_entering_yr2_ext`)。计算第2年净损益。
*   **最终指标**: 计算这两年净损益的NPV和总绝对损失。

### 2.4 策略3: 高风险可选12个月气球贷展期 (`simulate_strategy3_multiyear`)
*   **模拟周期**: `SIMULATION_YEARS` 年。
*   **状态跟踪**: 跟踪每个分箱中的贷款数量 (`count`) 和 **当前平均余额** (`balance`)。
*   **每年逻辑**:
    1.  年初应用 `CHURN_RATE` 减少各分箱贷款数 (`count`)。
    2.  计算年初组合价值 (基于当前 `count` 和 `balance`)，用于计算当年资金成本。
    3.  **分箱处理**:
        *   **低风险**:
            *   应用**当年**的基线气球贷逻辑 (根据当前 `balance` 计算利息和违约损失)。
            *   生存者进入下一年，`count` 更新，`balance` **不变**。
        *   **高风险**:
            *   **拒绝者** (`REFUSAL_PROBABILITY`): 按基线逻辑处理（基于当前 `balance` 计算利息或违约损失），损益计入当年。拒绝者**不进入**下一年。
            *   **接受者**:
                *   立即承受本金减免损失 (`initial_haircut_loss`)，计入当年损失。
                *   进入**新一轮**的12个月气球贷展期，本金为减免后 (`reduced_principal`)，利率为 `INTEREST_RATE_ANNUAL + EXTENSION_RATE_INCREASE`。
                *   展期内的年化PD使用**降低后**的PD (`pd_rate_annual * (1 - reduction_factor)`)。
                *   **展期损益**:
                    *   假设总违约的 1/3 发生在年中（不付息），2/3 发生在年末（付息）。
                    *   计算年中违约损失、年末违约损益（利息-损失）、生存者利息。
                    *   汇总展期内的损益和损失。
                *   **状态更新**: 只有展期生存者 (`expected_non_defaults`) 进入下一年，`count` 更新，`balance` 更新为**减免后的本金** (`reduced_principal`)。他们下一年**仍在高风险箱**，将再次面临策略3的逻辑（可能再次展期）。
    4.  **年计算**: 汇总当年所有损益和损失。计算当年资金成本。计算当年净损益。记录年末组合余额。
*   **最终指标**: 计算整个模拟期间年度净损益的NPV和总绝对损失，并记录年度组合余额变化。

## 3. 核心计算函数
*   `calculate_amortization_payment`: 计算分期摊还贷款的月供。
*   `calculate_outstanding_balance`: 计算分期摊还贷款在某时点的剩余本金。
*   `calculate_piecewise_survival_probs`: (策略2使用) 计算分段指数生存概率，校准到 `(1-annual_pd)^2` 的两年生存率，且前6个月风险率更高。
*   `calculate_npv`: 计算现金流序列的净现值(NPV)。

## 4. 输出
*   脚本会打印各策略不同参数下的最终NPV和总损失。
*   将年度 P&L(AF), 年度绝对损失, S3年度余额, S2/S3累计NPV 等详细结果保存到 `simulation_results` 目录下的 CSV 文件中。
*   生成对比图表 (NPV对比, 总损失对比, S3余额演变, 累计NPV演变)。

---
**请注意**: 此摘要基于对代码的理解，实际执行细节请以代码为准。特别是策略2和策略3的损益分配和状态更新逻辑较为复杂，建议仔细核对相关辅助函数和主循环逻辑。 