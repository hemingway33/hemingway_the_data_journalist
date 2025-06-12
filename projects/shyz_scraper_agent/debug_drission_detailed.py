#!/usr/bin/env python3
"""
详细调试DrissionPage版本的爬虫
观察浏览器中实际发生的情况和所有网络活动
"""

import time
import logging
from shyz_scraper_drission import SHYZScraperDrission, CompanyQuery

# 设置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_detailed_process():
    """详细调试查询过程"""
    print("=== 详细调试DrissionPage版本爬虫 ===")
    print("测试查询: 上海遇圆实业有限公司") 
    print("信用代码: 91310000MAE97Y0K5Y")
    print("观察浏览器行为和网络活动")
    print()
    
    # 创建爬虫实例 - 非headless模式，便于观察
    scraper = SHYZScraperDrission(
        headless=False,  # 显示浏览器
        timeout=60
    )
    
    try:
        print("1. 导航到查询页面...")
        success = scraper.navigate_to_query_page()
        
        if success:
            print("✅ 成功进入查询页面")
            
            # 暂停让用户观察页面状态
            input("按回车键继续到查询步骤...")
            
            print("2. 开始查询过程...")
            query = CompanyQuery(
                company_name="上海遇圆实业有限公司",
                credit_code="91310000MAE97Y0K5Y"
            )
            
            # 清空之前的网络请求记录
            scraper.network_requests.clear()
            
            print("3. 执行查询...")
            
            # 找到输入框
            company_input = scraper.page.ele("xpath://input[contains(@placeholder, '单位名称')]")
            code_input = scraper.page.ele("xpath://input[contains(@placeholder, '社会信用代码')]")
            query_button = scraper.page.ele("xpath://button[contains(text(), '查询')]")
            
            if company_input and code_input and query_button:
                print("✅ 找到所有必要元素")
                
                # 输入数据
                print("4. 输入查询数据...")
                js_code1 = f"arguments[0].value = '{query.company_name}'; arguments[0].dispatchEvent(new Event('input', {{bubbles: true}}));"
                scraper.page.run_js(js_code1, company_input)
                
                js_code2 = f"arguments[0].value = '{query.credit_code}'; arguments[0].dispatchEvent(new Event('input', {{bubbles: true}}));"
                scraper.page.run_js(js_code2, code_input)
                
                print(f"✅ 已输入公司名称: {company_input.value}")
                print(f"✅ 已输入信用代码: {code_input.value}")
                
                # 暂停让用户检查输入
                input("按回车键继续点击查询按钮...")
                
                print("5. 点击查询按钮...")
                query_button.click()
                print("✅ 查询按钮已点击")
                
                # 开始监控
                print("6. 开始监控查询结果...")
                print("   - 观察浏览器页面变化")
                print("   - 监听网络请求")
                print("   - 等待两阶段结果")
                print()
                
                start_time = time.time()
                stage1_detected = False
                
                for i in range(60):  # 60秒监控
                    current_time = time.time() - start_time
                    
                    # 检查网络请求
                    try:
                        responses = scraper.page.listen.responses
                        if responses and len(responses) > 0:
                            print(f"[{current_time:.1f}s] 发现 {len(responses)} 个网络响应")
                            
                            for j, response in enumerate(responses[-5:]):  # 只显示最新的5个
                                try:
                                    print(f"  响应 {j+1}: {response.method} {response.url}")
                                    if hasattr(response, 'status'):
                                        print(f"    状态: {response.status}")
                                    
                                    # 尝试获取响应内容
                                    try:
                                        body = response.body
                                        if body:
                                            body_str = str(body)
                                            if len(body_str) < 200:
                                                print(f"    内容: {body_str}")
                                            else:
                                                print(f"    内容长度: {len(body_str)} 字符")
                                                # 检查关键词
                                                keywords = ['印章', '公章', '备案', '无数据', '没有找到', '暂无']
                                                found_keywords = [kw for kw in keywords if kw in body_str]
                                                if found_keywords:
                                                    print(f"    包含关键词: {found_keywords}")
                                                    if any(kw in ['无数据', '没有找到', '暂无'] for kw in found_keywords):
                                                        stage1_detected = True
                                                        print(f"    ⚠️ 检测到第一阶段响应（无记录）")
                                                    if any(kw in ['印章', '公章', '备案'] for kw in found_keywords):
                                                        print(f"    ✅ 检测到印章相关数据！")
                                    except Exception as e:
                                        print(f"    获取响应内容失败: {e}")
                                        
                                except Exception as e:
                                    print(f"  处理响应时出错: {e}")
                    except Exception as e:
                        print(f"[{current_time:.1f}s] 获取网络响应失败: {e}")
                    
                    # 检查页面内容
                    try:
                        # 查找页面中的表格
                        tables = scraper.page.eles('xpath://table')
                        if tables:
                            print(f"[{current_time:.1f}s] 页面中有 {len(tables)} 个表格")
                            
                            for t_idx, table in enumerate(tables):
                                try:
                                    rows = table.eles('xpath:.//tr')
                                    if len(rows) > 1:
                                        print(f"  表格 {t_idx+1}: {len(rows)} 行")
                                        # 显示表格内容
                                        for r_idx, row in enumerate(rows[:3]):  # 只显示前3行
                                            cells = row.eles('xpath:.//td')
                                            if not cells:
                                                cells = row.eles('xpath:.//th')
                                            cell_texts = [cell.text.strip() for cell in cells]
                                            if any(cell_texts):
                                                print(f"    行 {r_idx+1}: {cell_texts}")
                                except Exception as e:
                                    print(f"  分析表格 {t_idx+1} 时出错: {e}")
                        
                        # 查找包含关键词的元素
                        for keyword in ['印章', '公章', '无数据', '没有找到']:
                            try:
                                elements = scraper.page.eles(f'xpath://*[contains(text(), "{keyword}")]')
                                if elements:
                                    print(f"[{current_time:.1f}s] 发现包含'{keyword}'的元素: {len(elements)}个")
                                    for elem in elements[:2]:  # 只显示前2个
                                        print(f"  内容: {elem.text.strip()}")
                            except Exception:
                                pass
                                
                    except Exception as e:
                        print(f"[{current_time:.1f}s] 检查页面内容失败: {e}")
                    
                    # 提示信息
                    if stage1_detected:
                        print(f"[{current_time:.1f}s] 第一阶段已检测到，继续等待第二阶段...")
                    elif i % 10 == 0 and i > 0:
                        print(f"[{current_time:.1f}s] 继续等待查询结果...")
                    
                    time.sleep(1)
                
                print("\n监控完成！")
                
                # 最终检查
                print("7. 最终检查页面状态...")
                try:
                    final_results = scraper._parse_page_results(query)
                    if final_results:
                        print(f"✅ 最终解析到 {len(final_results)} 条印章记录:")
                        for result in final_results:
                            print(f"  - {result.seal_name} ({result.seal_status})")
                    else:
                        print("⚠️ 最终页面解析未发现印章记录")
                except Exception as e:
                    print(f"❌ 最终页面解析失败: {e}")
                
                # 让用户手动检查
                input("\n请手动检查浏览器页面，确认是否有印章信息显示。按回车键继续...")
                
            else:
                print("❌ 无法找到必要的页面元素")
        else:
            print("❌ 无法进入查询页面")
            
    except Exception as e:
        print(f"❌ 调试过程中发生错误: {str(e)}")
        logger.exception("详细错误信息")
        
    finally:
        print("\n8. 关闭浏览器...")
        scraper.close()
        print("调试完成!")

if __name__ == "__main__":
    debug_detailed_process() 