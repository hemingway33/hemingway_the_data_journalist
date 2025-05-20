create table pcr_career_info
(
    id                  varchar(64)  not null comment '主键ID'
        primary key,
    order_id            varchar(64)  null comment '订单流水号',
    serial_id           varchar(64)  null comment '数据流水号',
    etl_id              varchar(64)  null comment '清洗流水号',
    report_id           varchar(32)  null comment '报告编号',
    employ_status_code  varchar(60)  null comment '就业状况',
    work_unit_name      varchar(128) null comment '工作单位',
    unit_code           varchar(60)  null comment '单位性质',
    industry_code       varchar(60)  null comment '行业',
    address             varchar(128) null comment '单位地址',
    tel                 varchar(32)  null comment '单位电话',
    occupation_code     varchar(60)  null comment '职业',
    post_code           varchar(60)  null comment '职务',
    title_code          varchar(60)  null comment '职称',
    entry_year          varchar(20)  null comment '进入本单位年份',
    updatetime_datetime datetime     null comment '信息更新日期',
    create_time         datetime     null comment '入库时间'
)
    comment 'B_05-职业信息' engine = InnoDB;

INSERT INTO zb_hongxi_datafactory.pcr_career_info (id, order_id, serial_id, etl_id, report_id, employ_status_code, work_unit_name, unit_code, industry_code, address, tel, occupation_code, post_code, title_code, entry_year, updatetime_datetime, create_time) VALUES ('20240919180803836708826263461889', 'lmw202409191807_01', '20240919180755836708792012640256', '20240919180800836708813659578368', '2023051911582406665908', '在职', 'XX科技有限公司', '私营企业', '信息技术', '北京市海淀区XX路1号', '13800138000', '软件工程师', '高级工程师', '无', '2015', '2024-09-19 10:08:03', '2024-09-19 10:08:03');
INSERT INTO zb_hongxi_datafactory.pcr_career_info (id, order_id, serial_id, etl_id, report_id, employ_status_code, work_unit_name, unit_code, industry_code, address, tel, occupation_code, post_code, title_code, entry_year, updatetime_datetime, create_time) VALUES ('20241022170905848652787576221697', 'lmw202410221708_01', '20241022170855848652745171808256', '20241022170902848652775836364800', '2023052315130550666501', '离职', 'YY教育集团', '国有企业', '教育', '上海市浦东新区XX路2号', '13900139000', '教师', '副校长', '高级教师', '2010', '2024-10-22 09:09:06', '2024-10-22 09:09:06');
INSERT INTO zb_hongxi_datafactory.pcr_career_info (id, order_id, serial_id, etl_id, report_id, employ_status_code, work_unit_name, unit_code, industry_code, address, tel, occupation_code, post_code, title_code, entry_year, updatetime_datetime, create_time) VALUES ('20241022170908848652799068614657', 'lmw202410221708_02', '20241022170901848652771356848128', '20241022170906848652791074271232', '2023052410224110665402', '在职', 'ZZ金融公司', '股份制企业', '金融', '广州市天河区XX路3号', '13700137000', '金融分析师', '部门经理', '无', '2018', '2024-10-22 09:09:08', '2024-10-22 09:09:08');
