create table pcr_credit_tips_list
(
    id                       varchar(64) not null comment '主键ID'
        primary key,
    order_id                 varchar(64) null comment '订单流水号',
    serial_id                varchar(64) null comment '数据流水号',
    etl_id                   varchar(64) null comment '清洗流水号',
    report_id                varchar(32) null comment '报告编号',
    credittips_business_code varchar(60) null comment '业务类型',
    business_class_code      varchar(60) null comment '业务大类',
    account_num              int         null comment '账户数',
    first_credit_month       varchar(20) null comment '首笔业务发放月份',
    fk_id                    varchar(32) null comment '外键FK_ID',
    create_time              datetime    null comment '入库时间'
)
    comment 'C_02_01-信贷交易提示信息_详情' engine = InnoDB;

INSERT INTO zb_hongxi_datafactory.pcr_credit_tips_list (id, order_id, serial_id, etl_id, report_id, credittips_business_code, business_class_code, account_num, first_credit_month, fk_id, create_time) VALUES ('20240919180803836708826263461889', 'lmw202409191807_01', '20240919180755836708792012640256', '20240919180800836708813659578368', '2023051911582406665908', null, null, null, null, null, '2024-09-19 10:08:03');
INSERT INTO zb_hongxi_datafactory.pcr_credit_tips_list (id, order_id, serial_id, etl_id, report_id, credittips_business_code, business_class_code, account_num, first_credit_month, fk_id, create_time) VALUES ('20241022170905848652787576221697', 'lmw202410221708_01', '20241022170855848652745171808256', '20241022170902848652775836364800', '2023052315130550666501', null, null, null, null, null, '2024-10-22 09:09:06');
INSERT INTO zb_hongxi_datafactory.pcr_credit_tips_list (id, order_id, serial_id, etl_id, report_id, credittips_business_code, business_class_code, account_num, first_credit_month, fk_id, create_time) VALUES ('20241022170908848652799068614657', 'lmw202410221708_02', '20241022170901848652771356848128', '20241022170906848652791074271232', '2023052410224110665402', null, null, null, null, null, '2024-10-22 09:09:08');
INSERT INTO zb_hongxi_datafactory.pcr_credit_tips_list (id, order_id, serial_id, etl_id, report_id, credittips_business_code, business_class_code, account_num, first_credit_month, fk_id, create_time) VALUES ('20241022193109848688541358632961', 'lmw202410221928_29', '20241022193101848688505338085376', '20241022193108848688537504067584', '2023051911582406665908', null, null, null, null, null, '2024-10-22 11:31:10');
