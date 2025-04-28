
好的，我们来用中文逐步解释一下 `simulate_two_strategies.py` 这个 Python 脚本是如何工作的，以及它是如何实现模拟和比较不同贷款组合退出策略的目标的。

**1. 初始化设置与参数定义 (大约第 5-48 行)**

*   **导入库:** 导入必要的库：`numpy` 用于数值计算，`matplotlib.pyplot` 和 `seaborn` 用于绘图，`pandas` 用于数据处理（尤其是 CSV 导出），`os` 用于与文件系统交互（创建结果目录）。
*   **定义核心参数:** 设置模拟的基本假设：
    *   投资组合规模 (`N_LOANS`)，平均贷款余额 (`AVG_BALANCE`)。
    *   贷款特征：年利率 (`INTEREST_RATE_ANNUAL`)，风险分箱数量 (`N_BINS`)，哪些分箱是高风险 (`HIGH_RISK_BINS`)。
    *   风险参数：每个分箱的预测违约率 (`PD_RATES`)，标准回收率 (`STANDARD_RECOVERY_RATE`)。
    *   特定策略参数：策略1测试的回收率 (`RECOVERY_RATES_STRAT1`)，策略2和3的违约降低系数 (`DEFAULT_REDUCTION_FACTORS`)，展期条款/细节 (`EXTENSION_MONTHS`, `PRINCIPAL_REDUCTION_FACTOR`, `REFUSAL_PROBABILITY`, `EXTENSION_RATE_INCREASE`)，以及策略2分段生存模型的风险率比率 (`HAZARD_RATE_RATIO`)。
    *   多年期模拟参数：年客户流失率 (`CHURN_RATE`)，模拟持续时间 (`SIMULATION_YEARS`)，NPV 的贴现率 (`DISCOUNT_RATE`)。
    *   基础计算：计算初始投资组合总价值 (`TOTAL_PORTFOLIO_VALUE`) 和原始月利息 (`MONTHLY_INTEREST_PAYMENT`, `TOTAL_INTEREST_ORIGINAL`) 作为参考（尽管多年期逻辑更动态）。

**2. 辅助函数定义 (大约第 51-148 行)**

*   **`calculate_npv(cash_flows, rate)`:** 接收一个年度现金流列表和一个贴现率，返回最终的净现值 (NPV) 和一个逐年累计 NPV 的列表。这对于跨时间比较策略的财务表现至关重要。
*   **`calculate_amortization_payment(...)`:** 标准的金融公式，用于计算分期摊还（随时间偿还本金和利息）贷款的固定月付款额。专门用于策略 2 的展期。
*   **`calculate_outstanding_balance(...)`:** 计算分期摊还贷款在一定数量的付款后剩余的本金。也用于策略 2。
*   **`calculate_exponential_survival_probs(...)`:** (现在较少使用，保留作参考) 假设违约风险率*恒定*（从年化 PD 推导），计算月度生存概率。
*   **`calculate_piecewise_survival_probs(...)`:** 计算月度生存概率，假设风险率是*分段恒定*的：前 6 个月有较高的比率 (`lambda1`)，之后有较低的比率 (`lambda2`)。计算 `lambda1` 和 `lambda2` 时，会保持一个特定的比率 (`HAZARD_RATE_RATIO`)，同时确保整体 2 年生存概率与输入的 `annual_pd` 所隐含的概率相匹配。此函数专门用于策略 2，以模拟观察到的违约在前 6 个月更集中的现象。

**3. 多年期模拟函数 (大约第 154-652 行)**

*   **`simulate_baseline_multiyear(...)`:**
    *   **目标:** 模拟“什么都不做”或标准运营情景，以提供基准。
    *   **过程:**
        *   初始化每个分箱的贷款数量。
        *   按 `SIMULATION_YEARS` 循环遍历每一年 `year`。
        *   **客户流失:** 在年初根据 `CHURN_RATE` 减少每个分箱的贷款数量。
        *   **模拟年度活动:** 对于每个分箱中剩余的贷款：
            *   根据分箱的 `PD_RATES` 计算预期违约。
            *   计算毛损益 (P&L)：未违约者的利息减去违约者的本金损失（使用 `STANDARD_RECOVERY_RATE`）。
            *   确定存续到下一年的未违约贷款数量。
        *   **融资成本:** 根据年初的投资组合价值和 `FUNDING_COST_RATE_ANNUAL` 计算成本。
        *   **净损益与状态更新:** 计算该年度融资后的净损益，记录年度损益和绝对损失，更新总绝对损失，并设置下一年度开始时的贷款数量。
        *   **最终指标:** 循环结束后，使用 `calculate_npv` 计算最终的总 NPV 和年度累计 NPV。
    *   **输出:** 返回最终 NPV、总绝对损失，以及年度累计 NPV、年度净损益和年度绝对损失的列表。

*   **`simulate_strategy1_multiyear(recovery_rate_recall, ...)`:**
    *   **目标:** 模拟策略 1，其中高风险贷款被以不同的方式召回/终止。
    *   **过程:** 结构与基线类似（初始化、年度循环、客户流失、融资成本、最终 NPV）。
    *   **关键区别:** 在年度活动模拟内部：
        *   低风险分箱：逻辑与基线相同。
        *   高风险分箱：仍然根据 PD 计算预期违约和未违约。但是，计算违约者的本金损失时使用传入的 `recovery_rate_recall` 而不是 `STANDARD_RECOVERY_RATE`。
    *   **输出:** 格式与基线相同。

*   **`_calculate_s2_low_risk_outcome(...)` & `_calculate_s2_high_risk_acceptor_outcome(...)`:** 这些是*专门为策略 2* 服务的*辅助*函数。它们封装了为一组贷款模拟 24 个月分期摊还展期期间的复杂逻辑，使用了*分段生存模型*。它们计算给定一组贷款在*整个 24 个月展期期间*产生的总损益 (P&L) 和损失 (Loss)。
*   **`simulate_strategy2_multiyear(default_reduction_factor, ...)`:**
    *   **目标:** 模拟策略 2（强制性 24 个月分期摊还展期）。注意：此模拟硬编码为仅运行 **2 年**，因为核心事件本身就是 24 个月的展期。
    *   **过程:**
        *   仅在最开始应用*一次*客户流失。
        *   遍历*初始*（流失后）的贷款分箱。
        *   **低风险分箱:** 调用 `_calculate_s2_low_risk_outcome` 获取这些贷款的*总* 24 个月损益/损失。将此总额平均分配到第 1 年和第 2 年的结果中。
        *   **高风险分箱:**
            *   将贷款分为“拒绝者”（按基线逻辑处理，损益/损失分配到第 1 年）和“接受者”。
            *   对于接受者，调用 `_calculate_s2_high_risk_acceptor_outcome`（它包含了本金削减的影响，并使用 `default_reduction_factor` 和分段生存模型）来获取*总* 24 个月损益/损失。将结果（根据本金削减损失的时间进行调整）分配到第 1 年和第 2 年。
        *   **融资成本:** 分别为第 1 年（基于进入展期的初始价值）和第 2 年（基于估计仍活跃的价值）计算。
        *   **净损益与最终指标:** 计算年度融资后净损益并汇总损失。根据这 2 年的现金流计算最终 NPV。
    *   **输出:** 格式与基线相同，但现金流/损失仅跨越 2 年。

*   **`simulate_strategy3_multiyear(default_reduction_factor, ...)`:**
    *   **目标:** 模拟策略 3（选择性展期，采用滚动的 12 个月气球贷条款）。
    *   **过程:**
        *   使用更复杂的状态表示 (`portfolio_state`) 来跟踪每个分箱的贷款 `count`（数量）和平均 `balance`（余额），因为余额会因本金削减而改变。
        *   按 `SIMULATION_YEARS` 循环遍历每一年 `year`。
        *   **客户流失:** 在年初应用。
        *   **模拟年度活动:**
            *   低风险分箱：遵循基线逻辑（12 个月气球贷，基于原始 PD 违约/还清）。幸存者以相同余额结转。
            *   高风险分箱：
                *   区分“拒绝者”（仅在本年度遵循基线逻辑）。
                *   “接受者”：应用本金削减（立即产生损失/影响损益），计算*削减后的本金*。使用*提高后的利率*和*降低后的 PD*（由于 `default_reduction_factor`）模拟一个 *12 个月的球贷期限*。计算此 12 个月展期期间的损益/损失。关键是，*幸存者*以*削减后的余额*结转到 `next_year_portfolio_state`，准备在下一年可能重复展期过程。
        *   **融资成本:** 根据年初的投资组合价值计算。
        *   **净损益与状态更新:** 记录年度结果，更新总损失，跟踪年末余额，并更新 `portfolio_state` 以供下一次迭代使用。
        *   **最终指标:** 计算最终 NPV 和累计 NPV。
    *   **输出:** 返回最终 NPV、总损失，以及年度年末余额、累计 NPV、净损益和绝对损失的列表。

**4. 主执行块 (`if __name__ == "__main__":`) (大约第 655 行 - 结尾)**

*   **流程调度:** 当直接执行脚本时运行此块。
    *   打印状态消息。
    *   运行一次 `simulate_baseline_multiyear` 获取基准结果。
    *   **策略 1 循环:** 遍历 `RECOVERY_RATES_STRAT1` 中的每个 `recovery_rate`，为每个比率调用 `simulate_strategy1_multiyear`，并将返回的 NPV、损失和年度细节存储在以回收率作为键的字典中。
    *   **策略 2 循环:** 遍历 `DEFAULT_REDUCTION_FACTORS` 中的每个 `reduction_factor`，调用 `simulate_strategy2_multiyear`，并将结果存储在以降低系数作为键的字典中。
    *   **策略 3 循环:** 遍历 `reduction_factor`，调用 `simulate_strategy3_multiyear`，并类似地存储结果。
*   **导出结果到 CSV (大约第 762-810 行):**
    *   如果 `simulation_results` 目录不存在，则创建它。
    *   定义辅助函数 `save_yearly_results` 以便轻松地将字典数据（键是参数，值是年度列表）保存为转置的 CSV 文件。
    *   将基线、S1、S2 和 S3 的年度损益和损失保存到单独的 CSV 文件中。
    *   创建一个汇总 DataFrame，包含每次模拟运行（基线、S1 不同回收率、S2 不同降低系数、S3 不同降低系数）的最终 NPV 和总损失，并将其保存到 `results_final_summary.csv`。
*   **可视化 (大约第 812 行 - 结尾):**
    *   **字体配置:** 设置 Matplotlib/Seaborn 以使用合适的字体来显示绘图标签/标题中的中文字符（这对翻译后的图表至关重要）。
    *   **生成 4 个图表:**
        *   **NPV 对比:** 两个子图。左侧显示 S1 NPV vs 回收率。右侧显示 S2 和 S3 NPV vs 违约降低系数。两者都包含基线 NPV 作为水平线以供比较。
        *   **绝对损失对比:** 结构与 NPV 图类似，但显示的是总绝对损失。
        *   **S3 余额演变:** 显示在策略 3 下，对于不同的降低系数，总投资组合余额如何逐年变化。
        *   **累计 NPV:** 绘制基线、S2（每个 RF 有多条线）和 S3（每个 RF 有多条线）的逐年累计 NPV。还在结束年份将最终 S1 NPV 的*范围*显示为垂直条。图例放在绘图区域之外。
    *   使用 Seaborn 以获得更好的美观效果 (`sns.set_theme`)。
    *   设置标题、标签（中文）、图例和网格线以提高清晰度。
    *   使用 `plt.tight_layout()` 或 `plt.subplots_adjust()` 来防止标签/标题重叠。
    *   调用 `plt.show()` 来显示生成的图表。

总而言之，该脚本系统地定义了参数，模拟了多种情景（基线和三个具有不同参数的不同策略），计算了关键财务指标（NPV、损失），将详细的年度和汇总结果导出到 CSV 文件以供外部分析，并生成了比较图表以直观地分析不同退出策略之间的权衡。
