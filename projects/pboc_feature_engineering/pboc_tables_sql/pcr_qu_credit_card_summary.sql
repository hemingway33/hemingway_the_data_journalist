create table pcr_qu_credit_card_summary
(
    id                                   varchar(64)    not null comment '主键ID'
        primary key,
    order_id                             varchar(64)    null comment '订单流水号',
    serial_id                            varchar(64)    null comment '数据流水号',
    etl_id                               varchar(64)    null comment '清洗流水号',
    report_id                            varchar(32)    null comment '报告编号',
    org_num                              int            null comment '发卡机构数',
    account_num                          int            null comment '账户数',
    qucreditcard_total_credit            decimal(12, 2) null comment '授信总额',
    qucreditcard_max_credit              decimal(12, 2) null comment '单家行最高授信额',
    qucreditcard_min_credit              decimal(12, 2) null comment '单家行最低授信额',
    used_balance                         decimal(12, 2) null comment '透支余额',
    qucreditcard_recent_6mm_used_balance decimal(12, 2) null comment '最近6个月平均透支余额',
    create_time                          datetime       null comment '入库时间'
)
    comment 'C_10-准贷记卡账户汇总信息' engine = InnoDB;

INSERT INTO zb_hongxi_datafactory.pcr_qu_credit_card_summary (id, order_id, serial_id, etl_id, report_id, org_num, account_num, qucreditcard_total_credit, qucreditcard_max_credit, qucreditcard_min_credit, used_balance, qucreditcard_recent_6mm_used_balance, create_time) VALUES ('20240919180803836708826263461889', 'lmw202409191807_01', '20240919180755836708792012640256', '20240919180800836708813659578368', '2023051911582406665908', 3, 5, 10000.00, 5000.00, 1000.00, 2000.00, 1500.00, '2024-09-19 10:08:03');
INSERT INTO zb_hongxi_datafactory.pcr_qu_credit_card_summary (id, order_id, serial_id, etl_id, report_id, org_num, account_num, qucreditcard_total_credit, qucreditcard_max_credit, qucreditcard_min_credit, used_balance, qucreditcard_recent_6mm_used_balance, create_time) VALUES ('20241022170905848652787576221697', 'lmw202410221708_01', '20241022170855848652745171808256', '20241022170902848652775836364800', '2023052315130550666501', 2, 3, 8000.00, 4000.00, 2000.00, 1000.00, 800.00, '2024-10-22 09:09:06');
INSERT INTO zb_hongxi_datafactory.pcr_qu_credit_card_summary (id, order_id, serial_id, etl_id, report_id, org_num, account_num, qucreditcard_total_credit, qucreditcard_max_credit, qucreditcard_min_credit, used_balance, qucreditcard_recent_6mm_used_balance, create_time) VALUES ('20241022170908848652799068614657', 'lmw202410221708_02', '20241022170901848652771356848128', '20241022170906848652791074271232', '2023052410224110665402', 4, 7, 12000.00, 6000.00, 1500.00, 3000.00, 2000.00, '2024-10-22 09:09:08');
