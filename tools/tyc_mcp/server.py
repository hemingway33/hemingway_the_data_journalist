from typing import Dict, List, Optional, Any
from mcp import McpServer, McpTool, McpResource
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
import time
import logging
import traceback
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tyc_mcp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TianyanchaMCP')

class TianyanchaConfig(BaseModel):
    """Configuration for Tianyancha API"""
    username: str
    password: str
    base_url: str = "https://www.tianyancha.com"
    log_level: str = "DEBUG"

    def __init__(self, **data):
        super().__init__(**data)
        # Set log level based on config
        logging.getLogger('TianyanchaMCP').setLevel(getattr(logging, self.log_level.upper()))

class CompanyBasicInfo(BaseModel):
    """Basic company information"""
    name: str
    registration_number: Optional[str]
    legal_representative: Optional[str]
    registered_capital: Optional[str]
    establishment_date: Optional[str]
    business_status: Optional[str]
    company_type: Optional[str]
    industry: Optional[str]
    registration_authority: Optional[str]
    address: Optional[str]

class LegalCase(BaseModel):
    """Legal case information"""
    case_number: str
    case_type: str
    filing_date: Optional[str]
    court: Optional[str]
    status: Optional[str]
    title: str

class BusinessRisk(BaseModel):
    """Business risk information"""
    risk_type: str
    risk_level: str
    risk_date: Optional[str]
    risk_description: str

class IntellectualProperty(BaseModel):
    """Intellectual property information"""
    ip_type: str
    name: str
    application_date: Optional[str]
    status: Optional[str]
    number: Optional[str]

class CompanyDevelopment(BaseModel):
    """Company development information"""
    event_type: str
    date: Optional[str]
    description: str
    amount: Optional[str]

class TianyanchaServer(McpServer):
    """MCP server for Tianyancha data access"""
    
    def __init__(self, config: TianyanchaConfig):
        super().__init__()
        self.config = config
        logger.info(f"Initializing TianyanchaServer with base URL: {config.base_url}")
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Origin": "https://www.tianyancha.com",
            "Referer": "https://www.tianyancha.com/login"
        })
        logger.debug("Session headers configured")
        self._login()

    def _login(self):
        """Login to Tianyancha"""
        try:
            logger.info("Starting login process")
            # First get the login page to get any necessary cookies
            login_page_url = urljoin(self.config.base_url, "/login")
            logger.debug(f"Fetching login page: {login_page_url}")
            response = self.session.get(login_page_url)
            response.raise_for_status()
            
            # Extract any necessary tokens or form data from the login page
            soup = BeautifulSoup(response.text, 'html.parser')
            logger.debug("Login page parsed successfully")
            
            # Prepare login data
            login_data = {
                "account_no": self.config.username,
                "password": self.config.password,
                "remember": "true"
            }
            logger.debug("Login data prepared")
            
            # Send login request
            login_url = urljoin(self.config.base_url, "/api/login")
            logger.debug(f"Sending login request to: {login_url}")
            response = self.session.post(
                login_url,
                json=login_data,
                headers={
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest"
                }
            )
            response.raise_for_status()
            
            # Check if login was successful
            result = response.json()
            if result.get("state") != "ok":
                error_msg = result.get("message", "Unknown error")
                logger.error(f"Login failed: {error_msg}")
                raise Exception(f"Login failed: {error_msg}")
            
            logger.info("Login successful")
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during login: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to login: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during login: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to login: {str(e)}")

    def _check_login_status(self):
        """Check if the session is still valid and re-login if necessary"""
        try:
            profile_url = urljoin(self.config.base_url, "/user/profile")
            logger.debug(f"Checking login status at: {profile_url}")
            response = self.session.get(profile_url)
            if response.status_code != 200:
                logger.warning("Login status check failed, attempting to re-login")
                self._login()
            else:
                logger.debug("Login status check successful")
        except Exception as e:
            logger.error(f"Error checking login status: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            self._login()

    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make a request with login status check"""
        try:
            logger.debug(f"Making {method} request to: {url}")
            logger.debug(f"Request parameters: {kwargs}")
            
            self._check_login_status()
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            logger.debug(f"Request successful with status code: {response.status_code}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise

    def _parse_basic_info(self, soup: BeautifulSoup) -> CompanyBasicInfo:
        """Parse basic company information from HTML"""
        try:
            logger.debug("Starting to parse basic company information")
            
            # Extract company name
            name = soup.find('h1', class_='name').text.strip() if soup.find('h1', class_='name') else None
            logger.debug(f"Found company name: {name}")
            
            # Extract registration number
            reg_number = None
            reg_number_elem = soup.find('div', text=re.compile(r'统一社会信用代码'))
            if reg_number_elem:
                reg_number = reg_number_elem.find_next('div').text.strip()
            logger.debug(f"Found registration number: {reg_number}")
            
            # Extract legal representative
            lr = None
            lr_elem = soup.find('div', text=re.compile(r'法定代表人'))
            if lr_elem:
                lr = lr_elem.find_next('div').text.strip()
            
            # Extract registered capital
            capital = None
            capital_elem = soup.find('div', text=re.compile(r'注册资本'))
            if capital_elem:
                capital = capital_elem.find_next('div').text.strip()
            
            # Extract establishment date
            est_date = None
            est_date_elem = soup.find('div', text=re.compile(r'成立日期'))
            if est_date_elem:
                est_date = est_date_elem.find_next('div').text.strip()
            
            # Extract business status
            status = None
            status_elem = soup.find('div', text=re.compile(r'经营状态'))
            if status_elem:
                status = status_elem.find_next('div').text.strip()
            
            # Extract company type
            company_type = None
            type_elem = soup.find('div', text=re.compile(r'企业类型'))
            if type_elem:
                company_type = type_elem.find_next('div').text.strip()
            
            # Extract industry
            industry = None
            industry_elem = soup.find('div', text=re.compile(r'所属行业'))
            if industry_elem:
                industry = industry_elem.find_next('div').text.strip()
            
            # Extract registration authority
            reg_auth = None
            reg_auth_elem = soup.find('div', text=re.compile(r'登记机关'))
            if reg_auth_elem:
                reg_auth = reg_auth_elem.find_next('div').text.strip()
            
            # Extract address
            address = None
            address_elem = soup.find('div', text=re.compile(r'企业地址'))
            if address_elem:
                address = address_elem.find_next('div').text.strip()
            
            logger.info(f"Successfully parsed basic info for company: {name}")
            return CompanyBasicInfo(
                name=name,
                registration_number=reg_number,
                legal_representative=lr,
                registered_capital=capital,
                establishment_date=est_date,
                business_status=status,
                company_type=company_type,
                industry=industry,
                registration_authority=reg_auth,
                address=address
            )
        except Exception as e:
            logger.error(f"Failed to parse basic info: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to parse basic info: {str(e)}")

    def _parse_legal_cases(self, soup: BeautifulSoup) -> List[LegalCase]:
        """Parse legal cases from HTML"""
        cases = []
        try:
            case_elements = soup.find_all('div', class_='case-item')
            for elem in case_elements:
                case = LegalCase(
                    case_number=elem.find('div', class_='case-number').text.strip(),
                    case_type=elem.find('div', class_='case-type').text.strip(),
                    filing_date=elem.find('div', class_='filing-date').text.strip(),
                    court=elem.find('div', class_='court').text.strip(),
                    status=elem.find('div', class_='status').text.strip(),
                    title=elem.find('div', class_='title').text.strip()
                )
                cases.append(case)
        except Exception as e:
            raise Exception(f"Failed to parse legal cases: {str(e)}")
        return cases

    def _parse_business_risks(self, soup: BeautifulSoup) -> List[BusinessRisk]:
        """Parse business risks from HTML"""
        risks = []
        try:
            risk_elements = soup.find_all('div', class_='risk-item')
            for elem in risk_elements:
                risk = BusinessRisk(
                    risk_type=elem.find('div', class_='risk-type').text.strip(),
                    risk_level=elem.find('div', class_='risk-level').text.strip(),
                    risk_date=elem.find('div', class_='risk-date').text.strip(),
                    risk_description=elem.find('div', class_='risk-description').text.strip()
                )
                risks.append(risk)
        except Exception as e:
            raise Exception(f"Failed to parse business risks: {str(e)}")
        return risks

    def _parse_intellectual_property(self, soup: BeautifulSoup) -> List[IntellectualProperty]:
        """Parse intellectual property from HTML"""
        ip_list = []
        try:
            ip_elements = soup.find_all('div', class_='ip-item')
            for elem in ip_elements:
                ip = IntellectualProperty(
                    ip_type=elem.find('div', class_='ip-type').text.strip(),
                    name=elem.find('div', class_='ip-name').text.strip(),
                    application_date=elem.find('div', class_='application-date').text.strip(),
                    status=elem.find('div', class_='ip-status').text.strip(),
                    number=elem.find('div', class_='ip-number').text.strip()
                )
                ip_list.append(ip)
        except Exception as e:
            raise Exception(f"Failed to parse intellectual property: {str(e)}")
        return ip_list

    def _parse_company_development(self, soup: BeautifulSoup) -> List[CompanyDevelopment]:
        """Parse company development events from HTML"""
        events = []
        try:
            event_elements = soup.find_all('div', class_='development-item')
            for elem in event_elements:
                event = CompanyDevelopment(
                    event_type=elem.find('div', class_='event-type').text.strip(),
                    date=elem.find('div', class_='event-date').text.strip(),
                    description=elem.find('div', class_='event-description').text.strip(),
                    amount=elem.find('div', class_='event-amount').text.strip()
                )
                events.append(event)
        except Exception as e:
            raise Exception(f"Failed to parse company development: {str(e)}")
        return events

    @McpTool("search_company")
    def search_company(self, keyword: str) -> List[CompanyBasicInfo]:
        """Search for companies by keyword"""
        try:
            logger.info(f"Searching for companies with keyword: {keyword}")
            response = self._make_request(
                "GET",
                urljoin(self.config.base_url, "/search"),
                params={"key": keyword}
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            companies = []
            
            # Parse search results
            company_elements = soup.find_all('div', class_='search-result-item')
            logger.debug(f"Found {len(company_elements)} company results")
            
            for elem in company_elements:
                try:
                    company = CompanyBasicInfo(
                        name=elem.find('div', class_='company-name').text.strip(),
                        registration_number=elem.find('div', class_='registration-number').text.strip(),
                        legal_representative=elem.find('div', class_='legal-representative').text.strip(),
                        registered_capital=elem.find('div', class_='registered-capital').text.strip(),
                        establishment_date=None,
                        business_status=elem.find('div', class_='business-status').text.strip(),
                        company_type=None,
                        industry=None,
                        registration_authority=None,
                        address=None
                    )
                    companies.append(company)
                    logger.debug(f"Parsed company: {company.name}")
                except Exception as e:
                    logger.warning(f"Failed to parse individual company: {str(e)}")
                    continue
            
            logger.info(f"Successfully parsed {len(companies)} companies")
            return companies
        except Exception as e:
            logger.error(f"Failed to search company: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to search company: {str(e)}")

    @McpTool("get_company_details")
    def get_company_details(self, company_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific company"""
        try:
            # Get basic info
            response = self._make_request(
                "GET",
                f"{self.config.base_url}/company/{company_id}"
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            basic_info = self._parse_basic_info(soup)
            
            # Get legal cases
            response = self._make_request(
                "GET",
                f"{self.config.base_url}/company/{company_id}/sifa"
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            legal_cases = self._parse_legal_cases(soup)
            
            # Get business risks
            response = self._make_request(
                "GET",
                f"{self.config.base_url}/company/{company_id}/jingxian"
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            business_risks = self._parse_business_risks(soup)
            
            # Get intellectual property
            response = self._make_request(
                "GET",
                f"{self.config.base_url}/company/{company_id}/intellectual-property"
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            ip_list = self._parse_intellectual_property(soup)
            
            # Get company development
            response = self._make_request(
                "GET",
                f"{self.config.base_url}/company/{company_id}/development"
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            development_events = self._parse_company_development(soup)
            
            return {
                "basic_info": basic_info.dict(),
                "legal_cases": [case.dict() for case in legal_cases],
                "business_risks": [risk.dict() for risk in business_risks],
                "intellectual_property": [ip.dict() for ip in ip_list],
                "development_events": [event.dict() for event in development_events]
            }
        except Exception as e:
            raise Exception(f"Failed to get company details: {str(e)}")

    @McpTool("get_company_legal_cases")
    def get_company_legal_cases(self, company_id: str) -> List[LegalCase]:
        """Get legal cases for a specific company"""
        try:
            response = self._make_request(
                "GET",
                f"{self.config.base_url}/company/{company_id}/sifa"
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._parse_legal_cases(soup)
        except Exception as e:
            raise Exception(f"Failed to get company legal cases: {str(e)}")

    @McpTool("get_company_business_risks")
    def get_company_business_risks(self, company_id: str) -> List[BusinessRisk]:
        """Get business risks for a specific company"""
        try:
            response = self._make_request(
                "GET",
                f"{self.config.base_url}/company/{company_id}/jingxian"
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._parse_business_risks(soup)
        except Exception as e:
            raise Exception(f"Failed to get company business risks: {str(e)}")

    @McpTool("get_company_intellectual_property")
    def get_company_intellectual_property(self, company_id: str) -> List[IntellectualProperty]:
        """Get intellectual property for a specific company"""
        try:
            response = self._make_request(
                "GET",
                f"{self.config.base_url}/company/{company_id}/intellectual-property"
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._parse_intellectual_property(soup)
        except Exception as e:
            raise Exception(f"Failed to get company intellectual property: {str(e)}")

    @McpTool("get_company_development")
    def get_company_development(self, company_id: str) -> List[CompanyDevelopment]:
        """Get company development events for a specific company"""
        try:
            response = self._make_request(
                "GET",
                f"{self.config.base_url}/company/{company_id}/development"
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._parse_company_development(soup)
        except Exception as e:
            raise Exception(f"Failed to get company development: {str(e)}")

    @McpResource("company_list")
    def get_company_list(self) -> List[Dict]:
        """Get a list of companies from recent searches"""
        # Implement company list resource
        return []

def main():
    # Example usage
    config = TianyanchaConfig(
        username="your_username",
        password="your_password",
        log_level="DEBUG"  # Set log level explicitly
    )
    logger.info("Starting Tianyancha MCP server")
    server = TianyanchaServer(config)
    server.run()

if __name__ == "__main__":
    main() 