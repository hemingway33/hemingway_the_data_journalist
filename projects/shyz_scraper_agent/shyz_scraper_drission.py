#!/usr/bin/env python3
"""
上海印章查询爬虫 - DrissionPage版本
使用接口监听方式获取查询结果，无需等待固定时间
"""

import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from DrissionPage import ChromiumPage, ChromiumOptions
from DrissionPage.common import Settings
import csv

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CompanyQuery:
    """公司查询信息"""
    company_name: str
    credit_code: str

@dataclass
class SealInfo:
    """印章信息"""
    company_name: str
    credit_code: str
    seal_name: str
    registration_date: str
    seal_status: str
    query_time: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

class SHYZScraperDrission:
    """上海印章查询爬虫 - DrissionPage版本"""
    
    def __init__(self, headless: bool = True, timeout: int = 30):
        """
        初始化爬虫
        
        Args:
            headless: 是否无头模式
            timeout: 超时时间(秒)
        """
        self.headless = headless
        self.timeout = timeout
        self.base_url = "https://gaj.sh.gov.cn/yzxt/"
        self.page = None
        self.network_requests = []  # 存储网络请求
        
        self._setup_browser()
    
    def _setup_browser(self):
        """设置浏览器"""
        try:
            # 配置ChromiumOptions
            co = ChromiumOptions()
            if self.headless:
                co.headless()
            
            # 添加反检测参数
            co.set_argument('--disable-blink-features=AutomationControlled')
            co.set_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            co.set_argument('--enable-network-service-logging')
            
            # 创建页面对象
            self.page = ChromiumPage(addr_or_opts=co)
            
            # 启动网络请求监听
            self._start_network_monitoring()
            
            logger.info("浏览器初始化完成")
            
        except Exception as e:
            logger.error(f"浏览器初始化失败: {e}")
            raise
    
    def _start_network_monitoring(self):
        """启动网络请求监听"""
        try:
            # 启动网络监听
            self.page.listen.start()
            logger.info("网络请求监听已启动")
            
        except Exception as e:
            logger.warning(f"启动网络监听失败: {e}")
    
    def navigate_to_query_page(self) -> bool:
        """导航到查询页面"""
        try:
            logger.info(f"正在访问首页: {self.base_url}")
            self.page.get(self.base_url)
            
            # 等待页面加载
            time.sleep(3)
            
            # 查找并点击信息查询链接 - 使用XPath
            info_query_selectors = [
                "xpath://div[@rel='indexXXquery']",
                "xpath://div[contains(text(), '信息查询')]",
                "xpath://a[contains(text(), '信息查询')]"
            ]
            
            info_query_element = None
            for selector in info_query_selectors:
                try:
                    info_query_element = self.page.ele(selector)
                    if info_query_element:
                        logger.info(f"找到信息查询元素，使用选择器: {selector}")
                        break
                except Exception as e:
                    logger.debug(f"选择器 {selector} 未找到元素: {e}")
                    continue
            
            if not info_query_element:
                logger.error("无法找到信息查询链接")
                return False
            
            # 点击信息查询
            logger.info("点击信息查询")
            info_query_element.click()
            
            # 等待查询页面加载
            time.sleep(3)
            
            # 验证是否进入查询页面
            query_form_selectors = [
                "xpath://input[contains(@placeholder, '单位名称')]",
                "xpath://input[contains(@placeholder, '企业名称')]",
                "xpath://label[contains(text(), '单位名称')]"
            ]
            
            query_form_found = False
            for selector in query_form_selectors:
                try:
                    element = self.page.ele(selector)
                    if element:
                        logger.info(f"查询表单已加载，找到元素: {selector}")
                        query_form_found = True
                        break
                except Exception:
                    continue
            
            if query_form_found:
                logger.info("成功导航到查询页面")
                return True
            else:
                logger.error("未找到查询表单，可能页面加载失败")
                return False
                
        except Exception as e:
            logger.error(f"导航到查询页面时发生错误: {e}")
            return False
    
    def search_company_seal(self, query: CompanyQuery) -> List[SealInfo]:
        """
        查询公司印章信息
        
        Args:
            query: 公司查询信息
            
        Returns:
            List[SealInfo]: 印章信息列表
        """
        try:
            logger.info(f"正在查询公司: {query.company_name} (信用代码: {query.credit_code})")
            
            # 清空之前的网络请求记录
            self.network_requests.clear()
            
            # 输入公司名称 - 使用真正正确的选择器
            company_input_selectors = [
                "css:#gzxxcx_dwmc",  # 真正被用户使用的输入框
                "xpath://input[@id='gzxxcx_dwmc']",
                "css:#main_getdwmc",  # 备用
                "xpath://input[@id='main_getdwmc']",
                "xpath://input[@name='yzdwmc']"
            ]
            
            company_input = None
            for selector in company_input_selectors:
                try:
                    company_input = self.page.ele(selector)
                    if company_input:
                        logger.info(f"找到单位名称输入框: {selector}")
                        break
                except Exception:
                    continue
            
            if not company_input:
                logger.error("无法找到单位名称输入框")
                return []
            
            # 使用简化的JavaScript方法输入公司名称
            logger.info("使用JavaScript方法输入公司名称")
            try:
                # 简化的JavaScript输入，基于调试发现这个方法有效
                js_code = f"""
                var input = arguments[0];
                var value = '{query.company_name}';
                
                // 聚焦并设置值
                input.focus();
                input.value = '';
                input.value = value;
                
                // 触发必要的事件
                input.dispatchEvent(new Event('input', {{bubbles: true}}));
                input.dispatchEvent(new Event('change', {{bubbles: true}}));
                
                return input.value;
                """
                
                result = self.page.run_js(js_code, company_input)
                time.sleep(0.5)
                
                # 验证输入结果
                current_value = (company_input.value or company_input.text or "").strip()
                if current_value == query.company_name:
                    logger.info(f"✅ 公司名称输入成功: {current_value}")
                    input_success = True
                else:
                    logger.error(f"❌ 公司名称输入失败，期望: '{query.company_name}', 实际: '{current_value}'")
                    input_success = False
                    
            except Exception as e:
                logger.error(f"JavaScript输入公司名称失败: {e}")
                input_success = False
            
            if not input_success:
                logger.error("公司名称输入失败，停止查询")
                return []
            
            # 输入社会信用代码 - 使用真正正确的选择器
            code_input_selectors = [
                "css:#gzxxcx_shxydm",  # 真正被用户使用的输入框
                "xpath://input[@id='gzxxcx_shxydm']",
                "css:#main_shxytybm",  # 备用
                "xpath://input[@id='main_shxytybm']",
                "xpath://input[contains(@placeholder, '社会信用代码')]",
                "xpath://input[contains(@placeholder, '信用代码')]"
            ]
            
            code_input = None
            for selector in code_input_selectors:
                try:
                    code_input = self.page.ele(selector)
                    if code_input:
                        logger.info(f"找到社会信用代码输入框: {selector}")
                        break
                except Exception:
                    continue
            
            if not code_input:
                logger.error("无法找到社会信用代码输入框")
                return []
            
            # 使用简化的JavaScript方法输入信用代码
            logger.info("使用JavaScript方法输入信用代码")
            try:
                # 简化的JavaScript输入，基于调试发现这个方法有效
                js_code = f"""
                var input = arguments[0];
                var value = '{query.credit_code}';
                
                // 聚焦并设置值
                input.focus();
                input.value = '';
                input.value = value;
                
                // 触发必要的事件
                input.dispatchEvent(new Event('input', {{bubbles: true}}));
                input.dispatchEvent(new Event('change', {{bubbles: true}}));
                
                return input.value;
                """
                
                result = self.page.run_js(js_code, code_input)
                time.sleep(0.5)
                
                # 验证输入结果
                current_code = (code_input.value or code_input.text or "").strip()
                if current_code == query.credit_code:
                    logger.info(f"✅ 信用代码输入成功: {current_code}")
                    code_input_success = True
                else:
                    logger.error(f"❌ 信用代码输入失败，期望: '{query.credit_code}', 实际: '{current_code}'")
                    code_input_success = False
                    
            except Exception as e:
                logger.error(f"JavaScript输入信用代码失败: {e}")
                code_input_success = False
            
            if not code_input_success:
                logger.error("信用代码输入失败，停止查询")
                return []
            
            # 点击查询按钮
            query_button_selectors = [
                "xpath://button[contains(text(), '查询')]",
                "xpath://input[@value='查询']",
                "xpath://a[contains(text(), '查询')]"
            ]
            
            query_button = None
            for selector in query_button_selectors:
                try:
                    query_button = self.page.ele(selector)
                    if query_button:
                        logger.info(f"找到查询按钮: {selector}")
                        break
                except Exception:
                    continue
            
            if not query_button:
                logger.error("无法找到查询按钮")
                return []
            
            # 最终验证输入内容
            final_company_name = (company_input.value or company_input.text or "").strip()
            final_credit_code = (code_input.value or code_input.text or "").strip()
            logger.info(f"查询前最终验证 - 公司名称: '{final_company_name}'")
            logger.info(f"查询前最终验证 - 信用代码: '{final_credit_code}'")
            
            # 严格验证：检查是否为空以及是否等于期望值
            if not final_company_name or not final_credit_code:
                logger.error("输入字段为空，无法进行查询")
                return []
            
            if final_company_name != query.company_name:
                logger.error(f"公司名称输入不匹配！期望: '{query.company_name}', 实际: '{final_company_name}'")
                logger.error("请手动检查输入是否正确")
                return []
                
            if final_credit_code != query.credit_code:
                logger.error(f"信用代码输入不匹配！期望: '{query.credit_code}', 实际: '{final_credit_code}'")
                logger.error("请手动检查输入是否正确")
                return []
                
            logger.info("✅ 输入验证通过，两个字段都正确匹配")
            
            logger.info("点击查询按钮")
            query_button.click()
            time.sleep(1)  # 等待查询请求发送
            
            # 等待网络请求完成并获取结果
            results = self._wait_for_query_results(query)
            
            return results
            
        except Exception as e:
            logger.error(f"查询公司印章信息时发生错误: {e}")
            return []
    
    def _wait_for_query_results(self, query: CompanyQuery, max_wait_time: int = 60) -> List[SealInfo]:
        """
        等待查询结果通过网络请求监听
        注意：网站采用两阶段查询机制：
        1. 第一阶段：立即返回"无查询记录"
        2. 第二阶段：几秒后返回真实结果（如果有的话）
        
        Args:
            query: 查询信息
            max_wait_time: 最大等待时间(秒) - 增加到60秒以等待两阶段结果
            
        Returns:
            List[SealInfo]: 印章信息列表
        """
        logger.info("等待查询结果（支持两阶段查询机制）...")
        start_time = time.time()
        
        stage1_completed = False  # 第一阶段是否完成
        stage2_wait_start = None  # 第二阶段等待开始时间
        
        while time.time() - start_time < max_wait_time:
            current_time = time.time() - start_time
            
            # 获取DrissionPage监听到的响应
            try:
                responses = self.page.listen.responses
                if responses:
                    logger.info(f"[{current_time:.1f}s] 发现 {len(responses)} 个网络响应")
                    
                    # 分析网络响应
                    for response in responses:
                        try:
                            url = response.url
                            # 检查是否是相关的查询请求
                            if any(keyword in url.lower() for keyword in ['query', 'search', 'yz', 'data', 'result', 'api']):
                                logger.info(f"[{current_time:.1f}s] 分析响应: {response.method} {url}")
                                
                                # 尝试获取响应体
                                try:
                                    response_body = response.body
                                    if response_body:
                                        response_str = str(response_body)
                                        
                                        # 检查是否包含"无记录"相关信息（第一阶段）
                                        if any(no_result in response_str for no_result in ['无数据', '没有找到', '暂无', '无记录', '查询不到']):
                                            logger.info(f"[{current_time:.1f}s] 检测到第一阶段响应（无记录提示）")
                                            stage1_completed = True
                                            stage2_wait_start = time.time()
                                        
                                        # 检查是否包含印章相关数据（第二阶段）
                                        elif any(keyword in response_str for keyword in ['印章', '公章', '备案', '登记']):
                                            logger.info(f"[{current_time:.1f}s] 检测到第二阶段响应（包含印章数据）")
                                            self.network_requests.append({
                                                'url': url,
                                                'method': response.method,
                                                'response': response_body,
                                                'timestamp': datetime.now().isoformat(),
                                                'stage': 'stage2'
                                            })
                                        else:
                                            # 记录所有相关响应以供分析
                                            self.network_requests.append({
                                                'url': url,
                                                'method': response.method,
                                                'response': response_body,
                                                'timestamp': datetime.now().isoformat(),
                                                'stage': 'unknown'
                                            })
                                            logger.debug(f"[{current_time:.1f}s] 记录响应数据，长度: {len(response_str)}")
                                            
                                except Exception as e:
                                    logger.debug(f"获取响应体失败: {e}")
                        except Exception as e:
                            logger.debug(f"处理单个响应失败: {e}")
                            
            except Exception as e:
                logger.debug(f"获取网络响应失败: {e}")
            
            # 检查页面DOM变化（重要：很多时候结果是通过DOM更新显示的）
            try:
                page_results = self._parse_page_results(query)
                if page_results:
                    logger.info(f"[{current_time:.1f}s] 从页面DOM中解析到印章数据")
                    return page_results
            except Exception as e:
                logger.debug(f"解析页面结果失败: {e}")
            
            # 分析收集到的网络数据
            if self.network_requests:
                try:
                    seal_infos = self._parse_network_responses(query)
                    if seal_infos:
                        logger.info(f"[{current_time:.1f}s] 从网络响应中解析到印章数据")
                        return seal_infos
                except Exception as e:
                    logger.debug(f"解析网络响应失败: {e}")
            
            # 两阶段查询逻辑
            if stage1_completed and stage2_wait_start:
                stage2_wait_time = time.time() - stage2_wait_start
                if stage2_wait_time < 15:  # 第二阶段等待最多15秒
                    if stage2_wait_time % 3 == 0:  # 每3秒记录一次等待状态
                        logger.info(f"[{current_time:.1f}s] 第一阶段已完成，等待第二阶段结果... ({stage2_wait_time:.1f}s)")
                else:
                    logger.info(f"[{current_time:.1f}s] 第二阶段等待超时，可能确实无记录")
                    break
            elif current_time > 10 and not stage1_completed:
                logger.info(f"[{current_time:.1f}s] 等待第一阶段响应...")
            
            time.sleep(1)  # 短暂等待再检查
        
        # 最终尝试解析页面
        try:
            final_page_results = self._parse_page_results(query)
            if final_page_results:
                logger.info("最终页面解析发现印章数据")
                return final_page_results
        except Exception as e:
            logger.debug(f"最终页面解析失败: {e}")
        
        # 输出调试信息
        if self.network_requests:
            logger.info(f"总共捕获到 {len(self.network_requests)} 个相关网络请求：")
            for i, req in enumerate(self.network_requests, 1):
                logger.info(f"  {i}. [{req.get('stage', 'unknown')}] {req['method']} {req['url']}")
        
        logger.info("查询完成，未获取到印章记录（可能该公司确实无印章登记）")
        return []
    
    def _parse_network_responses(self, query: CompanyQuery) -> List[SealInfo]:
        """
        解析网络响应中的印章数据
        
        Args:
            query: 查询信息
            
        Returns:
            List[SealInfo]: 印章信息列表
        """
        seal_infos = []
        
        for request in self.network_requests:
            try:
                response = request.get('response', {})
                
                # 处理JSON响应
                if isinstance(response, dict):
                    seal_infos.extend(self._extract_seals_from_json(response, query))
                
                # 处理HTML响应
                elif isinstance(response, str) and any(keyword in response for keyword in ['印章', '公章', '备案']):
                    seal_infos.extend(self._extract_seals_from_html(response, query))
                    
            except Exception as e:
                logger.warning(f"解析网络响应时出错: {e}")
                continue
        
        return seal_infos
    
    def _extract_seals_from_json(self, json_data: Dict, query: CompanyQuery) -> List[SealInfo]:
        """从JSON数据中提取印章信息"""
        seal_infos = []
        
        try:
            # 尝试不同的JSON结构
            data_fields = ['data', 'result', 'list', 'items', 'records']
            
            for field in data_fields:
                if field in json_data:
                    items = json_data[field]
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                seal_info = self._create_seal_info_from_dict(item, query)
                                if seal_info:
                                    seal_infos.append(seal_info)
                    elif isinstance(items, dict):
                        seal_info = self._create_seal_info_from_dict(items, query)
                        if seal_info:
                            seal_infos.append(seal_info)
            
            logger.info(f"从JSON中提取到 {len(seal_infos)} 条印章信息")
            
        except Exception as e:
            logger.warning(f"解析JSON数据时出错: {e}")
        
        return seal_infos
    
    def _extract_seals_from_html(self, html_content: str, query: CompanyQuery) -> List[SealInfo]:
        """从HTML内容中提取印章信息"""
        seal_infos = []
        
        try:
            # 使用DrissionPage解析HTML
            from DrissionPage import WebPage
            temp_page = WebPage()
            temp_page.html = html_content
            
            # 查找表格
            tables = temp_page.eles('tag:table')
            for table in tables:
                rows = table.eles('tag:tr')
                if len(rows) > 1:  # 至少有表头和数据行
                    # 解析表头
                    header_row = rows[0]
                    headers = [th.text.strip() for th in header_row.eles('tag:th')] or \
                             [td.text.strip() for td in header_row.eles('tag:td')]
                    
                    # 创建列映射
                    column_mapping = self._create_column_mapping(headers)
                    
                    # 解析数据行
                    for row in rows[1:]:
                        cells = row.eles('tag:td')
                        if len(cells) >= 3:
                            seal_info = self._create_seal_info_from_cells(cells, column_mapping, query)
                            if seal_info:
                                seal_infos.append(seal_info)
            
            logger.info(f"从HTML中提取到 {len(seal_infos)} 条印章信息")
            
        except Exception as e:
            logger.warning(f"解析HTML内容时出错: {e}")
        
        return seal_infos
    
    def _parse_page_results(self, query: CompanyQuery) -> List[SealInfo]:
        """
        解析当前页面的查询结果
        支持两阶段查询机制，检测动态加载的内容
        
        Args:
            query: 查询信息
            
        Returns:
            List[SealInfo]: 印章信息列表
        """
        seal_infos = []
        
        try:
            # 先检查是否有"无记录"提示
            no_result_indicators = [
                "xpath://*[contains(text(), '没有找到')]",
                "xpath://*[contains(text(), '无数据')]", 
                "xpath://*[contains(text(), '暂无')]",
                "xpath://*[contains(text(), '无记录')]",
                "xpath://*[contains(text(), '查询不到')]"
            ]
            
            has_no_result_message = False
            for selector in no_result_indicators:
                try:
                    elements = self.page.eles(selector)
                    for element in elements:
                        if element.text.strip():
                            logger.debug(f"发现无结果提示: {element.text.strip()}")
                            has_no_result_message = True
                            break
                    if has_no_result_message:
                        break
                except Exception:
                    continue
            
            # 查找结果表格 - 使用更广泛的选择器
            table_selectors = [
                "xpath://table",
                "xpath://div[contains(@class, 'table')]//table",
                "xpath://div[contains(@class, 'result')]//table",
                "xpath://div[contains(@id, 'result')]//table",
                "xpath://*[contains(@class, 'data')]//table"
            ]
            
            tables_found = []
            for selector in table_selectors:
                try:
                    tables = self.page.eles(selector)
                    tables_found.extend(tables)
                except Exception:
                    continue
            
            # 去重
            unique_tables = []
            for table in tables_found:
                if table not in unique_tables:
                    unique_tables.append(table)
            
            logger.debug(f"页面中发现 {len(unique_tables)} 个唯一表格")
            
            for table_idx, table in enumerate(unique_tables):
                try:
                    rows = table.eles('xpath:.//tr')
                    if len(rows) <= 1:  # 没有数据行
                        continue
                    
                    logger.debug(f"分析表格 {table_idx+1}，共 {len(rows)} 行")
                    
                    # 解析表头
                    header_row = rows[0]
                    ths = header_row.eles('xpath:.//th')
                    tds = header_row.eles('xpath:.//td')
                    headers = [th.text.strip() for th in ths] if ths else [td.text.strip() for td in tds]
                    
                    logger.debug(f"表格 {table_idx+1} 列标题: {headers}")
                    
                    # 检查是否是印章相关表格
                    header_text = ' '.join(headers).lower()
                    if not any(keyword in header_text for keyword in ['印章', '公章', '名称', '备案', '登记', '状态']):
                        logger.debug(f"表格 {table_idx+1} 不是印章相关表格，跳过")
                        continue
                    
                    logger.info(f"发现印章相关表格 {table_idx+1}，列标题: {headers}")
                    
                    # 创建列映射
                    column_mapping = self._create_column_mapping(headers)
                    logger.debug(f"列映射: {column_mapping}")
                    
                    # 解析数据行
                    for row_idx, row in enumerate(rows[1:], 1):
                        try:
                            cells = row.eles('xpath:.//td')
                            if len(cells) >= 1:  # 至少需要1列数据
                                # 获取所有单元格文本
                                cell_texts = [cell.text.strip() for cell in cells]
                                logger.debug(f"行 {row_idx} 数据: {cell_texts}")
                                
                                # 跳过空行或无效行
                                if not any(cell_texts) or all(text in ['-', '', '无'] for text in cell_texts):
                                    continue
                                
                                # 检查是否包含印章相关信息
                                row_text = ' '.join(cell_texts)
                                if not any(keyword in row_text for keyword in ['印章', '公章', '章']) and len(cell_texts[0]) < 3:
                                    continue
                                
                                seal_info = self._create_seal_info_from_cells(cells, column_mapping, query)
                                if seal_info:
                                    seal_infos.append(seal_info)
                                    logger.info(f"解析到印章 {len(seal_infos)}: {seal_info.seal_name} (状态: {seal_info.seal_status})")
                        except Exception as e:
                            logger.warning(f"解析表格 {table_idx+1} 第 {row_idx} 行时出错: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"解析表格 {table_idx+1} 时出错: {e}")
                    continue
            
            # 如果没有找到表格，尝试查找其他结构的数据
            if not seal_infos:
                logger.debug("未找到表格数据，尝试查找其他结构")
                
                # 查找包含印章信息的div或其他元素
                seal_divs = []
                div_selectors = [
                    "xpath://div[contains(text(), '印章')]",
                    "xpath://div[contains(text(), '公章')]",
                    "xpath://span[contains(text(), '印章')]",
                    "xpath://span[contains(text(), '公章')]",
                    "xpath://*[contains(@class, 'seal')]",
                    "xpath://*[contains(@class, 'stamp')]"
                ]
                
                for selector in div_selectors:
                    try:
                        elements = self.page.eles(selector)
                        seal_divs.extend(elements)
                    except Exception:
                        continue
                
                if seal_divs:
                    logger.info(f"发现 {len(seal_divs)} 个可能包含印章信息的元素")
                    for div in seal_divs:
                        logger.debug(f"印章相关元素内容: {div.text.strip()}")
            
            if seal_infos:
                logger.info(f"页面解析完成，共获取 {len(seal_infos)} 条印章信息")
            elif has_no_result_message:
                logger.debug("页面显示无查询记录")
            else:
                logger.debug("页面解析未发现印章数据")
            
        except Exception as e:
            logger.warning(f"解析页面结果时出错: {e}")
        
        return seal_infos
    
    def _create_column_mapping(self, headers: List[str]) -> Dict[str, int]:
        """创建列标题到索引的映射"""
        column_mapping = {}
        
        for i, header in enumerate(headers):
            if "公章名称" in header or "印章名称" in header or "名称" in header:
                column_mapping['seal_name'] = i
            elif "备案日期" in header or "登记日期" in header or "日期" in header:
                column_mapping['registration_date'] = i
            elif "状态" in header or "印章状态" in header:
                column_mapping['seal_status'] = i
        
        return column_mapping
    
    def _create_seal_info_from_cells(self, cells, column_mapping: Dict[str, int], query: CompanyQuery) -> Optional[SealInfo]:
        """从表格单元格创建印章信息"""
        try:
            # 获取单元格文本
            cell_texts = [cell.text.strip() for cell in cells]
            
            # 使用列映射或默认索引
            seal_name = ""
            registration_date = ""
            seal_status = ""
            
            if 'seal_name' in column_mapping:
                seal_name = cell_texts[column_mapping['seal_name']] if column_mapping['seal_name'] < len(cell_texts) else ""
            elif len(cell_texts) > 0:
                seal_name = cell_texts[0]
            
            if 'registration_date' in column_mapping:
                registration_date = cell_texts[column_mapping['registration_date']] if column_mapping['registration_date'] < len(cell_texts) else ""
            elif len(cell_texts) > 1:
                registration_date = cell_texts[1]
                
            if 'seal_status' in column_mapping:
                seal_status = cell_texts[column_mapping['seal_status']] if column_mapping['seal_status'] < len(cell_texts) else ""
            elif len(cell_texts) > 2:
                seal_status = cell_texts[2]
            
            # 跳过空行或表头行
            if not seal_name or seal_name in ["公章名称", "印章名称", "名称", "-", ""]:
                return None
            
            # 跳过无效行
            if "没有" in seal_name or seal_name == "无":
                return None
            
            logger.info(f"创建印章信息 - 名称: {seal_name}, 日期: {registration_date}, 状态: {seal_status}")
            
            return SealInfo(
                company_name=query.company_name,
                credit_code=query.credit_code,
                seal_name=seal_name,
                registration_date=registration_date,
                seal_status=seal_status,
                query_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            logger.warning(f"从单元格创建印章信息时出错: {e}")
            logger.debug(f"单元格数据: {[cell.text.strip() for cell in cells]}")
            return None
    
    def _create_seal_info_from_dict(self, data: Dict, query: CompanyQuery) -> Optional[SealInfo]:
        """从字典数据创建印章信息"""
        try:
            # 尝试不同的字段名
            seal_name = data.get('seal_name') or data.get('sealName') or data.get('name') or data.get('公章名称')
            registration_date = data.get('registration_date') or data.get('registrationDate') or data.get('date') or data.get('备案日期')
            seal_status = data.get('seal_status') or data.get('sealStatus') or data.get('status') or data.get('状态')
            
            if not seal_name:
                return None
            
            return SealInfo(
                company_name=query.company_name,
                credit_code=query.credit_code,
                seal_name=str(seal_name),
                registration_date=str(registration_date or ""),
                seal_status=str(seal_status or ""),
                query_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            logger.warning(f"从字典创建印章信息时出错: {e}")
            return None
    
    def batch_query(self, queries: List[CompanyQuery]) -> List[SealInfo]:
        """
        批量查询公司印章信息
        
        Args:
            queries: 查询信息列表
            
        Returns:
            List[SealInfo]: 所有印章信息列表
        """
        all_results = []
        
        # 首先导航到查询页面
        if not self.navigate_to_query_page():
            logger.error("无法进入查询页面，批量查询终止")
            return all_results
        
        for i, query in enumerate(queries, 1):
            logger.info(f"开始第 {i}/{len(queries)} 个查询")
            
            try:
                results = self.search_company_seal(query)
                all_results.extend(results)
                
                logger.info(f"第 {i} 个查询完成，获得 {len(results)} 条记录")
                
                # 在查询之间稍作停顿，避免请求过快
                if i < len(queries):
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"第 {i} 个查询失败: {e}")
                continue
        
        logger.info(f"批量查询完成，总共获得 {len(all_results)} 条印章记录")
        return all_results
    
    def save_results(self, seal_infos: List[SealInfo], output_format: str = "json", 
                    filename: Optional[str] = None) -> str:
        """
        保存查询结果
        
        Args:
            seal_infos: 印章信息列表
            output_format: 输出格式 ("json" 或 "csv")
            filename: 输出文件名
            
        Returns:
            str: 输出文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"seal_query_results_{timestamp}"
        
        if output_format.lower() == "json":
            output_file = f"{filename}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([seal.to_dict() for seal in seal_infos], f, 
                         ensure_ascii=False, indent=2)
        else:  # CSV
            output_file = f"{filename}.csv"
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if seal_infos:
                    writer = csv.DictWriter(f, fieldnames=seal_infos[0].to_dict().keys())
                    writer.writeheader()
                    for seal in seal_infos:
                        writer.writerow(seal.to_dict())
        
        logger.info(f"查询结果已保存到: {output_file}")
        return output_file
    
    def close(self):
        """关闭浏览器"""
        try:
            if self.page:
                self.page.quit()
                logger.info("浏览器已关闭")
        except Exception as e:
            logger.warning(f"关闭浏览器时出错: {e}")

# 便捷函数
def load_queries_from_csv(csv_file: str) -> List[CompanyQuery]:
    """从CSV文件加载查询信息"""
    queries = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query = CompanyQuery(
                    company_name=row['company_name'].strip(),
                    credit_code=row['credit_code'].strip()
                )
                queries.append(query)
        logger.info(f"从 {csv_file} 加载了 {len(queries)} 个查询")
    except Exception as e:
        logger.error(f"加载CSV文件失败: {e}")
    
    return queries

if __name__ == "__main__":
    # 示例用法
    scraper = SHYZScraperDrission(headless=False)
    
    try:
        # 单个查询示例
        query = CompanyQuery(
            company_name="上海遇圆实业有限公司",
            credit_code="91310000MAE97Y0K5Y"
        )
        
        if scraper.navigate_to_query_page():
            results = scraper.search_company_seal(query)
            
            if results:
                print(f"查询成功！找到 {len(results)} 条印章记录:")
                for i, seal in enumerate(results, 1):
                    print(f"{i}. {seal.seal_name} - {seal.seal_status}")
                
                # 保存结果
                scraper.save_results(results, "json")
                scraper.save_results(results, "csv")
            else:
                print("未找到印章记录")
        
    finally:
        scraper.close() 