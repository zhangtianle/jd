CREATE TABLE t_user (
  uid         INT,
  age         INT,
  sex         INT,
  active_date DATE,
  limit       FLOAT
);

CREATE TABLE t_order (
  uid       INT,
  buy_time   DATE,
  price     FLOAT,
  qty INT,
  cate_id      INT,
  discount FLOAT
);

CREATE TABLE t_loan (
  uid       DATETIME,
  loan_time   FLOAT,
  loan_amount        FLOAT,
  plannum     INT
);


LOAD DATA LOCAL INFILE '~/dataset/o2o/ccf_offline_stage1_train.csv'
INTO TABLE offline_train FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';

LOAD DATA LOCAL INFILE '~/dataset/o2o/ccf_offline_stage1_test_revised.csv'
INTO TABLE offline_test FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';

LOAD DATA LOCAL INFILE '~/dataset/o2o/ccf_online_stage1_train.csv'
INTO TABLE online_train FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';