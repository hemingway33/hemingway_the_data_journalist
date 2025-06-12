#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock Scraper for 实物印章备案查询 (Physical Seal Filing Inquiry)
模拟实物印章备案查询服务的爬虫

This mock scraper simulates the behavior of a Chinese government service
for querying physical seal registrations and filings.
"""

import json
import random
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib


@dataclass
class SealRecord:
    """实物印章备案记录 - Physical Seal Filing Record"""
    seal_id: str  # 印章编号
    company_name: str  # 企业名称
    company_code: str  # 企业代码
    seal_type: str  # 印章类型
    filing_date: str  # 备案日期
    status: str  # 状态
    authority: str  # 备案机关
    contact_person: str  # 联系人
    contact_phone: str  # 联系电话
    remarks: str  # 备注


class MockSealInquiryService:
    """模拟实物印章备案查询服务"""
    
    def __init__(self):
        self.base_url = "https://mock-gov-seal-service.cn"
        self.session_timeout = 30 * 60  # 30分钟会话超时
        self.mock_data = self._generate_mock_data()
        
    def _generate_mock_data(self) -> List[SealRecord]:
        """生成模拟数据"""
        companies = [
            "北京科技有限公司", "上海贸易股份有限公司", "深圳创新科技集团",
            "广州制造业有限公司", "杭州互联网科技公司", "成都软件开发有限责任公司",
            "西安电子科技企业", "南京新材料股份公司", "武汉生物技术有限公司",
            "重庆机械制造集团", "天津化工有限公司", "青岛海洋科技公司"
        ]
        
        seal_types = ["公章", "财务专用章", "合同专用章", "人事专用章", "法人代表章"]
        authorities = ["市场监督管理局", "工商行政管理局", "公安局治安管理处"]
        statuses = ["正常", "注销", "变更", "冻结"]
        
        records = []
        for i in range(50):
            company = random.choice(companies)
            filing_date = datetime.now() - timedelta(days=random.randint(1, 365))
            
            record = SealRecord(
                seal_id=f"SEAL{2020 + i//10}{i:04d}",
                company_name=company,
                company_code=f"{random.randint(100000000000000000, 999999999999999999)}",
                seal_type=random.choice(seal_types),
                filing_date=filing_date.strftime("%Y-%m-%d"),
                status=random.choice(statuses),
                authority=random.choice(authorities),
                contact_person=f"张{chr(0x4e00 + random.randint(0, 100))}",
                contact_phone=f"1{random.randint(3, 9)}{random.randint(100000000, 999999999)}",
                remarks=f"备案编号: BA{random.randint(100000, 999999)}"
            )
            records.append(record)
            
        return records
    
    def _simulate_network_delay(self):
        """模拟网络延迟"""
        time.sleep(random.uniform(0.5, 2.0))
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        timestamp = str(int(time.time()))
        random_str = str(random.randint(10000, 99999))
        return hashlib.md5((timestamp + random_str).encode()).hexdigest()[:16]


class SealInquiryScraper:
    """实物印章备案查询爬虫"""
    
    def __init__(self):
        self.service = MockSealInquiryService()
        self.session_id = None
        self.login_status = False
        
    def connect(self) -> Dict:
        """连接到服务"""
        print("正在连接到实物印章备案查询系统...")
        self.service._simulate_network_delay()
        
        self.session_id = self.service._generate_session_id()
        
        return {
            "status": "success",
            "message": "连接成功",
            "session_id": self.session_id,
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def login(self, username: str, password: str) -> Dict:
        """登录系统"""
        print(f"正在登录用户: {username}")
        self.service._simulate_network_delay()
        
        # 模拟登录验证
        if len(username) >= 4 and len(password) >= 6:
            self.login_status = True
            return {
                "status": "success",
                "message": "登录成功",
                "user_id": username,
                "permissions": ["查询", "导出"]
            }
        else:
            return {
                "status": "error",
                "message": "用户名或密码错误"
            }
    
    def search_by_company_name(self, company_name: str) -> Dict:
        """按企业名称查询"""
        if not self.login_status:
            return {"status": "error", "message": "请先登录"}
        
        print(f"正在查询企业: {company_name}")
        self.service._simulate_network_delay()
        
        results = []
        for record in self.service.mock_data:
            if company_name in record.company_name:
                results.append({
                    "seal_id": record.seal_id,
                    "company_name": record.company_name,
                    "company_code": record.company_code,
                    "seal_type": record.seal_type,
                    "filing_date": record.filing_date,
                    "status": record.status,
                    "authority": record.authority
                })
        
        return {
            "status": "success",
            "message": f"找到 {len(results)} 条记录",
            "total_count": len(results),
            "results": results[:10],  # 限制返回前10条
            "has_more": len(results) > 10
        }
    
    def search_by_seal_id(self, seal_id: str) -> Dict:
        """按印章编号查询"""
        if not self.login_status:
            return {"status": "error", "message": "请先登录"}
        
        print(f"正在查询印章编号: {seal_id}")
        self.service._simulate_network_delay()
        
        for record in self.service.mock_data:
            if record.seal_id == seal_id:
                return {
                    "status": "success",
                    "message": "查询成功",
                    "result": {
                        "seal_id": record.seal_id,
                        "company_name": record.company_name,
                        "company_code": record.company_code,
                        "seal_type": record.seal_type,
                        "filing_date": record.filing_date,
                        "status": record.status,
                        "authority": record.authority,
                        "contact_person": record.contact_person,
                        "contact_phone": record.contact_phone,
                        "remarks": record.remarks
                    }
                }
        
        return {
            "status": "error",
            "message": "未找到相关记录"
        }
    
    def search_by_company_code(self, company_code: str) -> Dict:
        """按企业代码查询"""
        if not self.login_status:
            return {"status": "error", "message": "请先登录"}
        
        print(f"正在查询企业代码: {company_code}")
        self.service._simulate_network_delay()
        
        results = []
        for record in self.service.mock_data:
            if record.company_code == company_code:
                results.append({
                    "seal_id": record.seal_id,
                    "company_name": record.company_name,
                    "seal_type": record.seal_type,
                    "filing_date": record.filing_date,
                    "status": record.status
                })
        
        return {
            "status": "success",
            "message": f"找到 {len(results)} 条记录",
            "results": results
        }
    
    def get_seal_details(self, seal_id: str) -> Dict:
        """获取印章详细信息"""
        if not self.login_status:
            return {"status": "error", "message": "请先登录"}
        
        print(f"正在获取印章详细信息: {seal_id}")
        self.service._simulate_network_delay()
        
        for record in self.service.mock_data:
            if record.seal_id == seal_id:
                return {
                    "status": "success",
                    "seal_details": {
                        "basic_info": {
                            "seal_id": record.seal_id,
                            "company_name": record.company_name,
                            "company_code": record.company_code,
                            "seal_type": record.seal_type
                        },
                        "filing_info": {
                            "filing_date": record.filing_date,
                            "authority": record.authority,
                            "status": record.status
                        },
                        "contact_info": {
                            "contact_person": record.contact_person,
                            "contact_phone": record.contact_phone
                        },
                        "additional_info": {
                            "remarks": record.remarks,
                            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                }
        
        return {
            "status": "error", 
            "message": "印章记录不存在"
        }
    
    def export_results(self, query_params: Dict) -> Dict:
        """导出查询结果"""
        if not self.login_status:
            return {"status": "error", "message": "请先登录"}
        
        print("正在准备导出数据...")
        self.service._simulate_network_delay()
        
        export_id = f"EXP{int(time.time())}"
        return {
            "status": "success",
            "message": "导出任务已创建",
            "export_id": export_id,
            "download_url": f"{self.service.base_url}/download/{export_id}",
            "expires_at": (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def logout(self) -> Dict:
        """退出登录"""
        print("正在退出登录...")
        self.login_status = False
        self.session_id = None
        
        return {
            "status": "success",
            "message": "已退出登录"
        }


def demo_usage():
    """演示使用方法"""
    print("=== 实物印章备案查询系统演示 ===\n")
    
    # 创建爬虫实例
    scraper = SealInquiryScraper()
    
    # 1. 连接系统
    print("1. 连接系统")
    result = scraper.connect()
    print(f"   结果: {result}")
    print()
    
    # 2. 登录
    print("2. 用户登录")
    result = scraper.login("admin", "password123")
    print(f"   结果: {result}")
    print()
    
    # 3. 按企业名称查询
    print("3. 按企业名称查询")
    result = scraper.search_by_company_name("科技")
    print(f"   结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    print()
    
    # 4. 按印章编号查询
    print("4. 按印章编号查询")
    if result["status"] == "success" and result["results"]:
        seal_id = result["results"][0]["seal_id"]
        detail_result = scraper.get_seal_details(seal_id)
        print(f"   结果: {json.dumps(detail_result, ensure_ascii=False, indent=2)}")
    print()
    
    # 5. 导出结果
    print("5. 导出结果")
    export_result = scraper.export_results({"company_name": "科技"})
    print(f"   结果: {export_result}")
    print()
    
    # 6. 退出登录
    print("6. 退出登录")
    logout_result = scraper.logout()
    print(f"   结果: {logout_result}")


if __name__ == "__main__":
    demo_usage()
