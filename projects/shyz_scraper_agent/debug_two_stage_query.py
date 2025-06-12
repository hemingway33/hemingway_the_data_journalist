#!/usr/bin/env python3
"""
调试两阶段查询过程的专用脚本
分析查询后页面内容的变化过程
"""

import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_two_stage_query():
    """调试两阶段查询过程"""
    
    # 配置Chrome选项
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(10)
    
    try:
        # 1. 访问首页
        logger.info("访问首页...")
        driver.get("https://gaj.sh.gov.cn/yzxt/")
        time.sleep(3)
        
        # 2. 点击信息查询
        logger.info("点击信息查询...")
        info_query_element = driver.find_element(By.XPATH, "//div[@rel='indexXXquery']")
        info_query_element.click()
        time.sleep(3)
        
        # 等待查询页面加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[contains(@placeholder, '单位名称')]"))
        )
        logger.info("查询页面已加载")
        
        # 3. 输入查询信息
        logger.info("输入查询信息...")
        
        # 输入公司名称
        company_input = driver.find_element(By.XPATH, "//label[contains(text(), '单位名称')]/..//input")
        company_input.clear()
        company_input.send_keys("上海遇圆实业有限公司")
        logger.info("已输入公司名称")
        
        # 输入信用代码
        code_input = driver.find_element(By.XPATH, "//label[contains(text(), '社会信用代码')]/..//input")
        code_input.clear()
        code_input.send_keys("91310000MAE97Y0K5Y")
        logger.info("已输入信用代码")
        
        # 4. 点击查询按钮
        logger.info("点击查询按钮...")
        query_button = driver.find_element(By.XPATH, "//button[contains(text(), '查询')]")
        query_button.click()
        logger.info("查询按钮已点击")
        
        # 5. 分阶段观察页面变化
        logger.info("=== 开始监控页面变化 ===")
        
        for stage in range(1, 6):  # 监控5个阶段，每阶段5秒
            logger.info(f"\n--- 第 {stage} 阶段 (查询后 {stage * 5} 秒) ---")
            time.sleep(5)
            
            # 截取页面内容
            page_source_snippet = driver.page_source[:2000] + "..." if len(driver.page_source) > 2000 else driver.page_source
            logger.info(f"页面源码片段长度: {len(driver.page_source)}")
            
            # 检查常见的"无结果"信息
            no_result_texts = ["没有找到", "无数据", "暂无", "没有匹配", "无查询记录", "无记录", "查询不到"]
            for text in no_result_texts:
                if text in driver.page_source:
                    logger.info(f"发现无结果提示: '{text}'")
            
            # 查找所有可见的表格
            tables = driver.find_elements(By.TAG_NAME, "table")
            logger.info(f"页面中表格数量: {len(tables)}")
            
            for i, table in enumerate(tables):
                try:
                    if table.is_displayed():
                        rows = table.find_elements(By.TAG_NAME, "tr")
                        if rows:
                            logger.info(f"表格 {i+1}: {len(rows)} 行")
                            # 打印表格内容的前几行
                            for j, row in enumerate(rows[:3]):  # 只显示前3行
                                cells = row.find_elements(By.TAG_NAME, "td") + row.find_elements(By.TAG_NAME, "th")
                                cell_texts = [cell.text.strip() for cell in cells]
                                if any(cell_texts):  # 如果有非空内容
                                    logger.info(f"  行 {j+1}: {cell_texts}")
                except Exception as e:
                    logger.warning(f"分析表格 {i+1} 时出错: {e}")
            
            # 查找包含"印章"、"公章"等关键词的元素
            seal_keywords = ["印章", "公章", "备案", "登记"]
            for keyword in seal_keywords:
                try:
                    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{keyword}')]")
                    if elements:
                        logger.info(f"发现包含'{keyword}'的元素: {len(elements)} 个")
                        for elem in elements[:3]:  # 只显示前3个
                            if elem.is_displayed() and elem.text.strip():
                                logger.info(f"  内容: {elem.text.strip()}")
                except Exception as e:
                    logger.warning(f"搜索关键词'{keyword}'时出错: {e}")
            
            # 查找任何新出现的内容
            try:
                all_text_elements = driver.find_elements(By.XPATH, "//*[text()]")
                recent_texts = []
                for elem in all_text_elements:
                    if elem.is_displayed() and elem.text.strip():
                        text = elem.text.strip()
                        if len(text) > 5 and len(text) < 100:  # 过滤太短或太长的文本
                            recent_texts.append(text)
                
                if recent_texts:
                    logger.info(f"页面可见文本数量: {len(recent_texts)}")
                    # 显示一些示例文本
                    for text in recent_texts[:5]:
                        logger.info(f"  文本示例: {text}")
                        
            except Exception as e:
                logger.warning(f"分析页面文本时出错: {e}")
        
        logger.info("\n=== 监控完成 ===")
        
        # 最终检查整个页面
        logger.info("=== 最终页面分析 ===")
        final_html = driver.page_source
        
        # 保存最终HTML到文件以便详细分析
        with open("final_query_result.html", "w", encoding="utf-8") as f:
            f.write(final_html)
        logger.info("最终页面源码已保存到 final_query_result.html")
        
        # 等待用户按键继续（可以手动检查页面）
        input("\n按回车键继续（你可以在浏览器中手动检查页面内容）...")
        
    except Exception as e:
        logger.error(f"调试过程中发生错误: {e}")
        logger.exception("详细错误信息")
        
    finally:
        driver.quit()
        logger.info("浏览器已关闭")

if __name__ == "__main__":
    debug_two_stage_query() 