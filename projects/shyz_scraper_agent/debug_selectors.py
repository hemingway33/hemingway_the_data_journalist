#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选择器调试脚本 - 检查实际页面结构
Debug script to inspect actual page structure and find correct selectors
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

def debug_page_structure():
    """调试页面结构，找到正确的选择器"""
    print("=== 页面结构调试 ===")
    
    # 设置浏览器
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print("正在访问页面...")
        driver.get("https://gaj.sh.gov.cn/yzxt/")
        time.sleep(5)  # 等待页面完全加载
        
        print(f"页面标题: {driver.title}")
        
        # 查找所有可能的信息查询相关元素
        print("\n=== 查找信息查询相关元素 ===")
        
        # 查找包含"信息"或"查询"的所有元素
        info_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '信息') or contains(text(), '查询')]")
        
        if info_elements:
            print(f"找到 {len(info_elements)} 个包含'信息'或'查询'的元素:")
            for i, elem in enumerate(info_elements[:10], 1):  # 只显示前10个
                try:
                    tag = elem.tag_name
                    text = elem.text.strip()
                    classes = elem.get_attribute('class')
                    onclick = elem.get_attribute('onclick')
                    href = elem.get_attribute('href')
                    
                    print(f"\n  {i}. 标签: {tag}")
                    print(f"     文本: '{text}'")
                    if classes:
                        print(f"     类名: {classes}")
                    if onclick:
                        print(f"     点击事件: {onclick}")
                    if href:
                        print(f"     链接: {href}")
                        
                except Exception as e:
                    print(f"     获取元素信息时出错: {e}")
        
        # 查找所有链接
        print("\n=== 查找所有链接 ===")
        links = driver.find_elements(By.TAG_NAME, "a")
        info_links = [link for link in links if "信息" in link.text or "查询" in link.text]
        
        if info_links:
            print(f"找到 {len(info_links)} 个包含'信息'或'查询'的链接:")
            for i, link in enumerate(info_links, 1):
                try:
                    text = link.text.strip()
                    href = link.get_attribute('href')
                    print(f"  {i}. 文本: '{text}' -> {href}")
                except:
                    pass
        
        # 查找所有按钮
        print("\n=== 查找所有按钮 ===")
        buttons = driver.find_elements(By.TAG_NAME, "button")
        divs_as_buttons = driver.find_elements(By.XPATH, "//div[contains(@class, 'btn') or contains(@onclick, '')]")
        
        all_buttons = buttons + divs_as_buttons
        info_buttons = [btn for btn in all_buttons if "信息" in btn.text or "查询" in btn.text]
        
        if info_buttons:
            print(f"找到 {len(info_buttons)} 个包含'信息'或'查询'的按钮:")
            for i, btn in enumerate(info_buttons, 1):
                try:
                    text = btn.text.strip()
                    onclick = btn.get_attribute('onclick')
                    classes = btn.get_attribute('class')
                    print(f"  {i}. 文本: '{text}'")
                    if onclick:
                        print(f"     点击事件: {onclick}")
                    if classes:
                        print(f"     类名: {classes}")
                except:
                    pass
        
        # 查找页面主要结构
        print("\n=== 页面主要结构 ===")
        try:
            # 查找主要容器
            main_containers = driver.find_elements(By.XPATH, "//div[@class] | //section[@class] | //main[@class]")
            print(f"找到 {len(main_containers)} 个主要容器")
            
            # 查找导航相关元素
            nav_elements = driver.find_elements(By.XPATH, "//nav | //div[contains(@class, 'nav')] | //ul[contains(@class, 'nav')]")
            print(f"找到 {len(nav_elements)} 个导航元素")
            
            # 查找菜单相关元素
            menu_elements = driver.find_elements(By.XPATH, "//*[contains(@class, 'menu')] | //*[contains(@class, 'sidebar')]")
            print(f"找到 {len(menu_elements)} 个菜单元素")
            
        except Exception as e:
            print(f"分析页面结构时出错: {e}")
        
        # 截图保存
        try:
            screenshot_path = "page_screenshot.png"
            driver.save_screenshot(screenshot_path)
            print(f"\n页面截图已保存为: {screenshot_path}")
        except Exception as e:
            print(f"保存截图时出错: {e}")
            
        # 保存页面源码
        try:
            with open("page_source.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("页面源码已保存为: page_source.html")
        except Exception as e:
            print(f"保存页面源码时出错: {e}")
        
        print("\n请在浏览器中手动检查页面，找到信息查询的入口...")
        input("按回车键继续...")
        
    except Exception as e:
        print(f"调试过程中发生错误: {e}")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    debug_page_structure() 