#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上海市公安局印章管理系统批量查询爬虫
Shanghai Police Seal Management System Batch Query Scraper

功能：批量查询公司印章信息
Input: 单位名称 + 统一社会信用代码
Output: 公司印章详细信息
"""

import time
import json
import csv
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shyz_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CompanyQuery:
    """公司查询信息数据类"""
    company_name: str
    credit_code: str


@dataclass
class SealInfo:
    """印章信息数据类"""
    company_name: str
    credit_code: str
    seal_name: str
    registration_date: str
    seal_status: str
    query_time: str


class SHYZScraper:
    """上海印章管理系统爬虫类"""
    
    def __init__(self, headless: bool = True, timeout: int = 30):
        """
        初始化爬虫
        
        Args:
            headless: 是否使用无头模式
            timeout: 默认超时时间（秒）
        """
        self.base_url = "https://gaj.sh.gov.cn/yzxt/"
        self.timeout = timeout
        self.session = None
        self.driver = None
        self.headless = headless
        
        # 设置请求会话
        self._setup_session()
        
    def _setup_session(self):
        """设置requests会话"""
        self.session = requests.Session()
        
        # 设置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

    def _setup_driver(self):
        """设置Selenium WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(self.timeout)

    def navigate_to_query_page(self) -> bool:
        """
        导航到查询页面
        
        Returns:
            bool: 是否成功导航到查询页面
        """
        try:
            if not self.driver:
                self._setup_driver()
                
            logger.info(f"正在访问首页: {self.base_url}")
            self.driver.get(self.base_url)
            
            # 等待页面加载
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # 根据页面源码分析，查找信息查询入口
            # 尝试多个可能的选择器
            info_query_selectors = [
                "//div[@rel='indexXXquery']",  # 根据页面源码发现的选择器
                "//*[contains(@onclick, 'indexFootLoad') and contains(@rel, 'indexXXquery')]",
                "//div[contains(@class, 'UIIndexFootImage') and contains(@onclick, 'indexFootLoad')]//parent::*[@rel='indexXXquery']",
                "//*[contains(text(), '信息查询')]",
                "//*[contains(text(), '信息') and contains(text(), '查询')]"
            ]
            
            info_query_element = None
            for selector in info_query_selectors:
                try:
                    info_query_element = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    logger.info(f"找到信息查询元素，使用选择器: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if not info_query_element:
                logger.error("未找到'信息查询'链接")
                return False
            
            # 点击信息查询
            logger.info("点击信息查询")
            info_query_element.click()
            time.sleep(3)
            
            # 等待查询页面加载，查找公章信息查询或相关输入框
            # 查找单位名称输入框作为判断是否成功进入查询页面的标志
            company_name_selectors = [
                "//input[@placeholder='鼎程（上海）金融']",  # 根据截图中的placeholder
                "//input[contains(@placeholder, '单位名称')]",
                "//input[@name='companyName']",
                "//input[@id='companyName']",
                "//label[contains(text(), '单位名称')]/following-sibling::input",
                "//label[contains(text(), '单位名称')]/..//input"
            ]
            
            # 等待查询页面的表单元素出现
            query_form_loaded = False
            for selector in company_name_selectors:
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, selector))
                    )
                    query_form_loaded = True
                    logger.info(f"查询表单已加载，找到输入框: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if query_form_loaded:
                logger.info("成功导航到查询页面")
                return True
            else:
                logger.warning("点击信息查询后未找到查询表单，可能需要进一步导航")
                
                # 尝试查找并点击"公章信息查询"
                seal_query_selectors = [
                    "//a[contains(text(), '公章信息查询')]",
                    "//div[contains(text(), '公章信息查询')]", 
                    "//span[contains(text(), '公章信息查询')]",
                    "//*[contains(text(), '公章') and contains(text(), '查询')]",
                    "//div[contains(@class, 'UIIndexMenudiv') and contains(@onclick, 'ywhtml')][@rel='ywcx']"  # 根据页面源码
                ]
                
                seal_query_element = None
                for selector in seal_query_selectors:
                    try:
                        seal_query_element = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                        logger.info(f"找到公章查询元素，使用选择器: {selector}")
                        break
                    except TimeoutException:
                        continue
                
                if seal_query_element:
                    # 点击公章信息查询
                    logger.info("点击公章信息查询")
                    seal_query_element.click()
                    time.sleep(3)
                    
                    # 再次检查查询表单是否加载
                    for selector in company_name_selectors:
                        try:
                            WebDriverWait(self.driver, 10).until(
                                EC.presence_of_element_located((By.XPATH, selector))
                            )
                            logger.info("成功导航到查询页面")
                            return True
                        except TimeoutException:
                            continue
                
                logger.error("未能成功导航到查询页面")
                return False
                
        except Exception as e:
            logger.error(f"导航到查询页面时发生错误: {str(e)}")
            return False

    def search_company_seal(self, query: CompanyQuery) -> Optional[List[SealInfo]]:
        """
        查询单个公司的印章信息
        
        Args:
            query: 公司查询信息
            
        Returns:
            List[SealInfo]: 印章信息列表，失败返回None
        """
        try:
            if not self.driver:
                logger.error("WebDriver未初始化")
                return None
                
            logger.info(f"正在查询公司: {query.company_name} (信用代码: {query.credit_code})")
            
            # 等待页面完全加载
            time.sleep(2)
            
            # 查找单位名称输入框
            company_name_selectors = [
                "//input[contains(@placeholder, '单位名称') or contains(@placeholder, '鼎程')]",
                "//input[@placeholder='鼎程（上海）金融']",  # 根据截图中的placeholder
                "//input[contains(@placeholder, '单位名称')]",
                "//input[@name='companyName']",
                "//input[@id='companyName']",
                "//label[contains(text(), '单位名称')]/following-sibling::input",
                "//label[contains(text(), '单位名称')]/..//input"
            ]
            
            company_name_input = None
            for selector in company_name_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            company_name_input = element
                            logger.info(f"找到可交互的单位名称输入框: {selector}")
                            break
                    if company_name_input:
                        break
                except Exception as e:
                    logger.debug(f"尝试选择器 {selector} 时出错: {e}")
                    continue
            
            if not company_name_input:
                logger.error("未找到可交互的单位名称输入框")
                return None
            
            # 滚动到元素位置并确保可见
            self.driver.execute_script("arguments[0].scrollIntoView(true);", company_name_input)
            time.sleep(1)
            
            # 输入单位名称
            try:
                company_name_input.clear()
                company_name_input.send_keys(query.company_name)
                logger.info(f"已输入单位名称: {query.company_name}")
            except Exception as e:
                logger.error(f"输入单位名称时出错: {e}")
                # 尝试使用JavaScript输入
                try:
                    self.driver.execute_script("arguments[0].value = arguments[1];", company_name_input, query.company_name)
                    logger.info(f"使用JavaScript输入单位名称: {query.company_name}")
                except Exception as e2:
                    logger.error(f"JavaScript输入也失败: {e2}")
                    return None
            
            # 查找社会信用代码输入框
            credit_code_selectors = [
                "//input[contains(@placeholder, '社会信用代码') or contains(@placeholder, '91310104071162')]",
                "//input[@placeholder='91310104071162']",  # 根据截图中的placeholder
                "//input[contains(@placeholder, '社会信用代码')]",
                "//input[contains(@placeholder, '信用代码')]",
                "//input[@name='creditCode']",
                "//input[@id='creditCode']",
                "//label[contains(text(), '社会信用代码')]/following-sibling::input",
                "//label[contains(text(), '社会信用代码')]/..//input"
            ]
            
            credit_code_input = None
            for selector in credit_code_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            credit_code_input = element
                            logger.info(f"找到可交互的社会信用代码输入框: {selector}")
                            break
                    if credit_code_input:
                        break
                except Exception as e:
                    logger.debug(f"尝试选择器 {selector} 时出错: {e}")
                    continue
            
            if not credit_code_input:
                logger.error("未找到可交互的社会信用代码输入框")
                return None
            
            # 滚动到元素位置并确保可见
            self.driver.execute_script("arguments[0].scrollIntoView(true);", credit_code_input)
            time.sleep(1)
            
            # 输入社会信用代码
            try:
                credit_code_input.clear()
                credit_code_input.send_keys(query.credit_code)
                logger.info(f"已输入社会信用代码: {query.credit_code}")
            except Exception as e:
                logger.error(f"输入社会信用代码时出错: {e}")
                # 尝试使用JavaScript输入
                try:
                    self.driver.execute_script("arguments[0].value = arguments[1];", credit_code_input, query.credit_code)
                    logger.info(f"使用JavaScript输入社会信用代码: {query.credit_code}")
                except Exception as e2:
                    logger.error(f"JavaScript输入也失败: {e2}")
                    return None
            
            # 查找并点击查询按钮
            search_btn_selectors = [
                "//button[contains(text(), '查询')]",
                "//input[@value='查询']",
                "//a[contains(text(), '查询')]",
                "//*[@type='submit']",
                "//button[@type='submit']"
            ]
            
            search_btn = None
            for selector in search_btn_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            search_btn = element
                            logger.info(f"找到可交互的查询按钮: {selector}")
                            break
                    if search_btn:
                        break
                except Exception as e:
                    logger.debug(f"尝试选择器 {selector} 时出错: {e}")
                    continue
            
            if not search_btn:
                logger.error("未找到可交互的查询按钮")
                return None
            
            # 滚动到按钮位置
            self.driver.execute_script("arguments[0].scrollIntoView(true);", search_btn)
            time.sleep(1)
            
            # 点击查询按钮
            try:
                search_btn.click()
                logger.info("点击查询按钮")
            except Exception as e:
                logger.error(f"点击查询按钮时出错: {e}")
                # 尝试使用JavaScript点击
                try:
                    self.driver.execute_script("arguments[0].click();", search_btn)
                    logger.info("使用JavaScript点击查询按钮")
                except Exception as e2:
                    logger.error(f"JavaScript点击也失败: {e2}")
                    return None
            
            # 等待结果加载
            time.sleep(5)
            
            # 解析查询结果
            seal_infos = self._parse_seal_results(query)
            
            return seal_infos
            
        except Exception as e:
            logger.error(f"查询公司 {query.company_name} 时发生错误: {str(e)}")
            return None

    def _parse_seal_results(self, query: CompanyQuery) -> List[SealInfo]:
        """
        解析印章查询结果
        
        注意：网站查询机制是两阶段的：
        1. 先显示"无查询记录"
        2. 如果有数据，会在几秒后显示真实结果
        
        Args:
            query: 原始查询信息
            
        Returns:
            List[SealInfo]: 印章信息列表
        """
        seal_infos = []
        
        try:
            # 第一阶段：等待初始结果加载
            logger.info("等待查询结果加载（第一阶段）...")
            time.sleep(3)
            
            # 检查是否有"没有找到匹配的记录"类似的提示
            no_result_found = self._check_no_results()
            
            if no_result_found:
                logger.info("第一阶段显示无查询记录，等待可能的真实结果...")
                # 等待更长时间，看是否有真实结果出现
                time.sleep(8)  # 等待真实结果加载
                
                # 再次检查结果
                no_result_still = self._check_no_results()
                if no_result_still:
                    # 再等一次，确保没有遗漏
                    logger.info("继续等待真实结果...")
                    time.sleep(5)
            
            # 解析表格结果
            seal_infos = self._extract_table_results(query)
            
            if not seal_infos:
                logger.info("最终确认：未找到印章信息")
            else:
                logger.info(f"成功获取到 {len(seal_infos)} 条印章信息")
                
        except Exception as e:
            logger.error(f"解析查询结果时发生错误: {str(e)}")
            
        return seal_infos
    
    def _check_no_results(self) -> bool:
        """
        检查是否显示"无查询记录"信息
        
        Returns:
            bool: 是否显示无结果信息
        """
        try:
            no_result_selectors = [
                "//*[contains(text(), '没有找到')]",
                "//*[contains(text(), '无数据')]", 
                "//*[contains(text(), '暂无')]",
                "//*[contains(text(), '没有匹配')]",
                "//*[contains(text(), '无查询记录')]",
                "//*[contains(text(), '无记录')]",
                "//*[contains(text(), '查询不到')]"
            ]
            
            for selector in no_result_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    for element in elements:
                        if element.is_displayed() and element.text.strip():
                            logger.debug(f"找到无结果提示: {element.text.strip()}")
                            return True
                except Exception:
                    continue
                    
            return False
            
        except Exception as e:
            logger.error(f"检查无结果状态时出错: {e}")
            return False
    
    def _extract_table_results(self, query: CompanyQuery) -> List[SealInfo]:
        """
        从表格中提取印章结果
        
        Args:
            query: 查询信息
            
        Returns:
            List[SealInfo]: 印章信息列表
        """
        seal_infos = []
        
        try:
            # 查找结果表格
            table_selectors = [
                "//table",
                "//div[@class='table-responsive']//table",
                "//*[contains(@class, 'table')]",
                "//table[.//th or .//td]"  # 确保是有内容的表格
            ]
            
            results_table = None
            for selector in table_selectors:
                try:
                    tables = self.driver.find_elements(By.XPATH, selector)
                    for table in tables:
                        if table.is_displayed():
                            # 检查表格是否有内容
                            rows = table.find_elements(By.TAG_NAME, "tr")
                            if len(rows) > 1:  # 至少有表头和一行数据
                                results_table = table
                                logger.info(f"找到有内容的结果表格，共 {len(rows)} 行")
                                break
                    if results_table:
                        break
                except Exception:
                    continue
            
            if not results_table:
                logger.info("未找到包含数据的结果表格")
                return seal_infos
            
            # 解析表头，动态识别列
            header_row = results_table.find_elements(By.TAG_NAME, "tr")[0]
            headers = [th.text.strip() for th in header_row.find_elements(By.TAG_NAME, "th")]
            if not headers:
                headers = [td.text.strip() for td in header_row.find_elements(By.TAG_NAME, "td")]
            
            logger.info(f"表格列标题: {headers}")
            
            # 创建列索引映射
            column_mapping = {}
            for i, header in enumerate(headers):
                if "公章名称" in header or "印章名称" in header:
                    column_mapping['seal_name'] = i
                elif "备案日期" in header or "登记日期" in header or "日期" in header:
                    column_mapping['registration_date'] = i  
                elif "状态" in header or "印章状态" in header:
                    column_mapping['seal_status'] = i
            
            logger.info(f"列映射: {column_mapping}")
            
            # 查找数据行（跳过表头）
            data_rows = results_table.find_elements(By.TAG_NAME, "tr")[1:]
            
            if not data_rows:
                logger.info("表格中没有数据行")
                return seal_infos
            
            logger.info(f"开始解析 {len(data_rows)} 行数据")
            
            for row_idx, row in enumerate(data_rows):
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 3:  # 至少需要3列数据
                        # 使用动态列映射或默认索引
                        seal_name = cells[column_mapping.get('seal_name', 0)].text.strip()
                        registration_date = cells[column_mapping.get('registration_date', 1)].text.strip()
                        seal_status = cells[column_mapping.get('seal_status', 2)].text.strip()
                        
                        # 跳过空行或无效行
                        if not seal_name or seal_name == "-" or "没有" in seal_name:
                            continue
                            
                        seal_info = SealInfo(
                            company_name=query.company_name,
                            credit_code=query.credit_code,
                            seal_name=seal_name,
                            registration_date=registration_date,
                            seal_status=seal_status,
                            query_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        seal_infos.append(seal_info)
                        logger.info(f"解析到印章 {row_idx+1}: {seal_name} (状态: {seal_status})")
                        
                except Exception as e:
                    logger.warning(f"解析第 {row_idx+1} 行时出错: {str(e)}")
                    continue
                    
            logger.info(f"成功解析到 {len(seal_infos)} 条有效印章信息")
            
        except Exception as e:
            logger.error(f"提取表格结果时发生错误: {str(e)}")
            
        return seal_infos

    def batch_query(self, queries: List[CompanyQuery], output_file: str = None, 
                   delay: float = 2.0) -> List[SealInfo]:
        """
        批量查询公司印章信息
        
        Args:
            queries: 查询列表
            output_file: 输出文件路径（可选）
            delay: 查询间隔时间（秒）
            
        Returns:
            List[SealInfo]: 所有印章信息列表
        """
        all_seal_infos = []
        failed_queries = []
        
        logger.info(f"开始批量查询，共 {len(queries)} 个公司")
        
        # 首先导航到查询页面
        if not self.navigate_to_query_page():
            logger.error("无法导航到查询页面，批量查询终止")
            return all_seal_infos
        
        for i, query in enumerate(queries, 1):
            try:
                logger.info(f"进度: {i}/{len(queries)} - 查询 {query.company_name}")
                
                # 如果不是第一次查询，需要清空之前的输入
                if i > 1:
                    try:
                        # 清空输入框
                        company_input = self.driver.find_element(By.XPATH, "//input[contains(@placeholder, '单位名称') or contains(@placeholder, '鼎程')]")
                        company_input.clear()
                        
                        credit_input = self.driver.find_element(By.XPATH, "//input[contains(@placeholder, '社会信用代码') or contains(@placeholder, '91310104071162')]")
                        credit_input.clear()
                    except NoSuchElementException:
                        pass
                
                seal_infos = self.search_company_seal(query)
                
                if seal_infos:
                    all_seal_infos.extend(seal_infos)
                    logger.info(f"成功查询到 {len(seal_infos)} 条印章信息")
                else:
                    failed_queries.append(query)
                    logger.warning(f"未查询到 {query.company_name} 的印章信息")
                
                # 延迟，避免请求过于频繁
                if i < len(queries):
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"查询 {query.company_name} 时发生错误: {str(e)}")
                failed_queries.append(query)
                continue
        
        logger.info(f"批量查询完成，成功: {len(all_seal_infos)} 条，失败: {len(failed_queries)} 个")
        
        # 输出结果到文件
        if output_file and all_seal_infos:
            self._save_results(all_seal_infos, output_file)
        
        # 输出失败的查询
        if failed_queries:
            self._save_failed_queries(failed_queries, "failed_queries.json")
        
        return all_seal_infos

    def _save_results(self, seal_infos: List[SealInfo], output_file: str):
        """保存查询结果到文件"""
        try:
            if output_file.endswith('.csv'):
                # 保存为CSV
                df = pd.DataFrame([vars(info) for info in seal_infos])
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
            elif output_file.endswith('.json'):
                # 保存为JSON
                data = [vars(info) for info in seal_infos]
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                # 默认保存为JSON
                output_file += '.json'
                data = [vars(info) for info in seal_infos]
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
            logger.info(f"查询结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存结果时发生错误: {str(e)}")

    def _save_failed_queries(self, failed_queries: List[CompanyQuery], filename: str):
        """保存失败的查询到文件"""
        try:
            data = [vars(query) for query in failed_queries]
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"失败查询已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存失败查询时发生错误: {str(e)}")

    def close(self):
        """关闭浏览器和会话"""
        if self.driver:
            self.driver.quit()
        if self.session:
            self.session.close()


def load_queries_from_csv(csv_file: str) -> List[CompanyQuery]:
    """
    从CSV文件加载查询列表
    
    CSV格式：company_name,credit_code
    
    Args:
        csv_file: CSV文件路径
        
    Returns:
        List[CompanyQuery]: 查询列表
    """
    queries = []
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        for _, row in df.iterrows():
            query = CompanyQuery(
                company_name=str(row['company_name']).strip(),
                credit_code=str(row['credit_code']).strip()
            )
            queries.append(query)
            
        logger.info(f"从 {csv_file} 加载了 {len(queries)} 个查询")
        
    except Exception as e:
        logger.error(f"加载CSV文件时发生错误: {str(e)}")
        
    return queries


def main():
    """主函数 - 示例用法"""
    
    # 示例：创建查询列表
    queries = [
        CompanyQuery(
            company_name="上海遇圆实业有限公司",
            credit_code="91310000MAE97Y0K5Y"
        )

    ]
    
    # 或者从CSV文件加载
    # queries = load_queries_from_csv("companies.csv")
    
    # 创建爬虫实例
    scraper = SHYZScraper(headless=False)  # 设置为False以查看浏览器操作
    
    try:
        # 批量查询（不需要登录）
        results = scraper.batch_query(
            queries=queries,
            output_file="seal_results.csv",
            delay=2.0
        )
        
        print(f"查询完成，共获得 {len(results)} 条印章信息")
        
        # 打印部分结果
        for i, seal_info in enumerate(results[:5]):  # 只打印前5条
            print(f"\n印章信息 {i+1}:")
            print(f"  公司名称: {seal_info.company_name}")
            print(f"  信用代码: {seal_info.credit_code}")
            print(f"  印章名称: {seal_info.seal_name}")
            print(f"  备案日期: {seal_info.registration_date}")
            print(f"  印章状态: {seal_info.seal_status}")
            print(f"  查询时间: {seal_info.query_time}")
            
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {str(e)}")
    finally:
        # 关闭浏览器
        scraper.close()


if __name__ == "__main__":
    main()
