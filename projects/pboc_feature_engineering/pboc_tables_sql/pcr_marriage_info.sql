create table pcr_marriage_info
(
    id                  varchar(64)  not null comment '主键ID'
        primary key,
    order_id            varchar(64)  null comment '订单流水号',
    serial_id           varchar(64)  null comment '数据流水号',
    etl_id              varchar(64)  null comment '清洗流水号',
    report_id           varchar(32)  null comment '报告编号',
    marital_status_code varchar(60)  null comment '婚姻状况',
    spouse_name         varchar(32)  null comment '配偶姓名',
    spouse_id_code      varchar(60)  null comment '配偶证件类型',
    spouse_id_card_no   varchar(32)  null comment '配偶证件号码',
    spouse_work_unit    varchar(128) null comment '配偶工作单位',
    spouse_phone        varchar(32)  null comment '配偶联系电话',
    create_time         datetime     null comment '入库时间'
)
    comment 'B_03-婚姻信息' engine = InnoDB;

INSERT INTO zb_hongxi_datafactory.pcr_marriage_info (id, order_id, serial_id, etl_id, report_id, marital_status_code, spouse_name, spouse_id_code, spouse_id_card_no, spouse_work_unit, spouse_phone, create_time) VALUES ('20240919180803836708826263461889', 'lmw202409191807_01', '20240919180755836708792012640256', '20240919180800836708813659578368', '2023051911582406665908', '已婚', '李四', '身份证', '352625197809113334', 'XX公司', '13800138000', '2024-09-19 10:08:03');
INSERT INTO zb_hongxi_datafactory.pcr_marriage_info (id, order_id, serial_id, etl_id, report_id, marital_status_code, spouse_name, spouse_id_code, spouse_id_card_no, spouse_work_unit, spouse_phone, create_time) VALUES ('20241022170905848652787576221697', 'lmw202410221708_01', '20241022170855848652745171808256', '20241022170902848652775836364800', '2023052315130550666501', '未婚', '', '', '', '', '', '2024-10-22 09:09:06');
INSERT INTO zb_hongxi_datafactory.pcr_marriage_info (id, order_id, serial_id, etl_id, report_id, marital_status_code, spouse_name, spouse_id_code, spouse_id_card_no, spouse_work_unit, spouse_phone, create_time) VALUES ('20241022170908848652799068614657', 'lmw202410221708_02', '20241022170901848652771356848128', '20241022170906848652791074271232', '2023052410224110665402', '已婚', '王五', '护照', 'G12345678', 'YY公司', '13900139000', '2024-10-22 09:09:08');
