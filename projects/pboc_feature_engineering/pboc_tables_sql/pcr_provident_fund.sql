create table pcr_provident_fund
(
    id                  varchar(64)    not null comment '主键ID'
        primary key,
    order_id            varchar(64)    null comment '订单流水号',
    serial_id           varchar(64)    null comment '数据流水号',
    etl_id              varchar(64)    null comment '清洗流水号',
    report_id           varchar(32)    null comment '报告编号',
    pay_address         varchar(60)    null comment '参缴地',
    pay_datetime        datetime       null comment '参缴日期',
    pay_status_code     varchar(60)    null comment '缴费状态',
    begin_pay_mm        varchar(20)    null comment '初缴月份',
    end_pay_mm          varchar(20)    null comment '缴至月份',
    company_proportion  int            null comment '单位缴存比例',
    person_proportion   int            null comment '个人缴存比例',
    mm_deposit          decimal(12, 2) null comment '月缴存额',
    pay_company         varchar(240)   null comment '缴费单位',
    updatetime_datetime datetime       null comment '信息更新日期',
    create_time         datetime       null comment '入库时间'
)
    comment 'F_05-住房公积金参缴记录信息' engine = InnoDB;

INSERT INTO zb_hongxi_datafactory.pcr_provident_fund (id, order_id, serial_id, etl_id, report_id, pay_address, pay_datetime, pay_status_code, begin_pay_mm, end_pay_mm, company_proportion, person_proportion, mm_deposit, pay_company, updatetime_datetime, create_time) VALUES ('20240919180803836708826263461889', 'lmw202409191807_01', '20240919180755836708792012640256', '20240919180800836708813659578368', '2023051911582406665908', '北京市', '2024-09-19 10:08:03', '正常', '2023-01', '2024-09', 12, 8, 2000.00, 'XX公司', '2024-09-19 10:08:03', '2024-09-19 10:08:03');
INSERT INTO zb_hongxi_datafactory.pcr_provident_fund (id, order_id, serial_id, etl_id, report_id, pay_address, pay_datetime, pay_status_code, begin_pay_mm, end_pay_mm, company_proportion, person_proportion, mm_deposit, pay_company, updatetime_datetime, create_time) VALUES ('20241022170905848652787576221697', 'lmw202410221708_01', '20241022170855848652745171808256', '20241022170902848652775836364800', '2023052315130550666501', '上海市', '2024-10-22 09:09:06', '暂停', '2023-06', '2024-08', 10, 7, 1500.00, 'YY公司', '2024-10-22 09:09:06', '2024-10-22 09:09:06');
INSERT INTO zb_hongxi_datafactory.pcr_provident_fund (id, order_id, serial_id, etl_id, report_id, pay_address, pay_datetime, pay_status_code, begin_pay_mm, end_pay_mm, company_proportion, person_proportion, mm_deposit, pay_company, updatetime_datetime, create_time) VALUES ('20241022170908848652799068614657', 'lmw202410221708_02', '20241022170901848652771356848128', '20241022170906848652791074271232', '2023052410224110665402', '广州市', '2024-10-22 09:09:08', '正常', '2023-07', '2024-09', 11, 9, 2500.00, 'ZZ公司', '2024-10-22 09:09:08', '2024-10-22 09:09:08');
