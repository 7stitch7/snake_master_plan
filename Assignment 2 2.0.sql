DROP TABLE IF EXISTS prefers CASCADE;
DROP TABLE IF EXISTS owns CASCADE;
DROP TABLE IF EXISTS portfolio_action CASCADE;
DROP TABLE IF EXISTS manages CASCADE;
DROP TABLE IF EXISTS closing_price CASCADE;
DROP TABLE IF EXISTS MutualFund CASCADE;
DROP TABLE IF EXISTS location CASCADE;
DROP TABLE IF EXISTS Company CASCADE;
DROP TABLE IF EXISTS deposit_trans CASCADE;
DROP TABLE IF EXISTS Administrator CASCADE;
DROP TABLE IF EXISTS Customer CASCADE;
DROP TABLE IF EXISTS "User" CASCADE;
DROP TABLE IF EXISTS Deposit_Transaction CASCADE;
DROP TABLE IF EXISTS Investment_Transaction CASCADE;
DROP TABLE IF EXISTS Transaction CASCADE;



/* create the Company */
CREATE TABLE Company (
  company_id    INTEGER,
  name          VARCHAR(20) NOT NULL,
  CEO_fname     VARCHAR(10) NOT NULL,
  CEO_lname     VARCHAR(10) NOT NULL,
  PRIMARY KEY (company_id)
);
CREATE TABLE location (
  company_id    INTEGER,
  city        	Char(30)  NOT NULL,
  state 	VARCHAR(20) DEFAULT 'NSW',
  postcode      VARCHAR(10) NOT NULL,
  PRIMARY KEY (company_id),
  FOREIGN KEY (company_id) REFERENCES Company(company_id)
				ON DELETE CASCADE
);

/* create the MutualFund */
CREATE TABLE MutualFund (
  symbol 	VARCHAR(50) PRIMARY KEY,
  name  	VARCHAR(20) NOT NULL,
  c_date 	DATE NOT NULL,
  t_num_shares 	INTEGER NOT NULL,
  category 	VARCHAR(50) NOT NULL,
  description 	VARCHAR(50)
);

CREATE TABLE closing_price (
  p_date 	DATE,
  price  	DOUBLE PRECISION NOT NULL,
  symbol	VARCHAR(50) NOT NULL,
  UNIQUE (p_date, symbol),
  PRIMARY KEY (p_date, symbol),
  FOREIGN KEY (symbol) REFERENCES MutualFund (symbol)
				ON DELETE CASCADE
);

/* create the User */
CREATE TABLE "User" (
  login		VARCHAR(20),
  password	VARCHAR(20) NOT NULL,
  address	VARCHAR(50),
  email		VARCHAR(20),
  name		VARCHAR(20) NOT NULL,
  PRIMARY KEY (login)
);
CREATE TABLE Customer (
  login 	VARCHAR(20),
  balance	DOUBLE PRECISION,
  PRIMARY KEY (login),
  FOREIGN KEY (login) REFERENCES  "User"( login)
				ON DELETE CASCADE
);
CREATE TABLE Administrator (
  login 	VARCHAR(20),
  PRIMARY KEY (login),
  FOREIGN KEY (login) REFERENCES  "User"( login)
				ON DELETE CASCADE
);
/* create the Transaction */
CREATE TABLE Transaction(
  trans_id     INTEGER,
  sub_type     VARCHAR(50) CHECK (sub_type IN ('Deposit_Transaction','Investment_Transaction')),
  UNIQUE (trans_id,sub_type),
  PRIMARY KEY (trans_id)
);
CREATE TABLE Deposit_Transaction (
  trans_id	INTEGER PRIMARY KEY,
  sub_type     VARCHAR(50) CHECK (sub_type IN ('Deposit_Transaction')),
  FOREIGN KEY (trans_id,sub_type) REFERENCES  Transaction(trans_id,sub_type)
				ON DELETE CASCADE
				ON UPDATE NO ACTION
);
CREATE TABLE Investment_Transaction (
  trans_id	INTEGER PRIMARY KEY,
  sub_type     VARCHAR(50) CHECK (sub_type IN ('Investment_Transaction')),
  FOREIGN KEY (trans_id,sub_type) REFERENCES  Transaction(trans_id,sub_type)
				ON DELETE CASCADE
				ON UPDATE NO ACTION
  );

/* create the relationship*/
CREATE TABLE manages (
  symbol	VARCHAR(50),
  company_id    INTEGER NOT NULL,
  PRIMARY KEY (symbol),
  FOREIGN KEY (symbol) REFERENCES MutualFund(symbol)
				ON DELETE CASCADE,
  FOREIGN KEY (company_id) REFERENCES Company(company_id)
				ON DELETE CASCADE
);
CREATE TABLE deposit_trans (
  trans_id	INTEGER,
  login 	VARCHAR(20) NOT NULL,
  t_date  	DATE NOT NULL,
  amount	DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (trans_id),
  FOREIGN KEY (trans_id) REFERENCES Deposit_Transaction(trans_id)
				ON DELETE CASCADE,
  FOREIGN KEY (login) REFERENCES Customer(login)
				ON DELETE CASCADE
);
CREATE TABLE prefers (
  login		VARCHAR(20) PRIMARY KEY,
  symbol	VARCHAR(50) NOT NULL,
  percentage 	FLOAT NOT NULL CHECK(percentage <= 1 AND percentage > 0),
  FOREIGN KEY (login) REFERENCES Customer(login)
				ON DELETE CASCADE,
  FOREIGN KEY (symbol) REFERENCES MutualFund(symbol)
				ON DELETE CASCADE
);
CREATE TABLE owns(
  login		VARCHAR(20) PRIMARY KEY,
  symbol	VARCHAR(50) NOT NULL,
  shares 	INTEGER NOT NULL,
  FOREIGN KEY (login) REFERENCES Customer(login)
				ON DELETE CASCADE,
  FOREIGN KEY (symbol) REFERENCES MutualFund(symbol)
				ON DELETE CASCADE
);
CREATE TABLE portfolio_action (
  trans_id 	INTEGER PRIMARY KEY,
  login 	VARCHAR(20),
  symbol	VARCHAR(50),
  action	VARCHAR(50) CHECK (action IN ('sell','buy')),
  t_date	DATE,
  amount	DOUBLE PRECISION CHECK (amount =  price * num_shares),	
  price		DOUBLE PRECISION,
  num_shares	INTEGER,
  FOREIGN KEY (trans_id) REFERENCES Investment_Transaction(trans_id)
				ON DELETE CASCADE,
  FOREIGN KEY (login) REFERENCES Customer(login)
				ON DELETE CASCADE,
  FOREIGN KEY (symbol) REFERENCES MutualFund(symbol)
				ON DELETE CASCADE
);

/*
*This is the test command
*/

/* insert values into talbe Company */
INSERT INTO Company (company_id, name, CEO_fname, CEO_lname)
VALUES (00001, 'TESLA', 'ELLON', 'MASK');

/* insert values into talbe location */
INSERT INTO location (company_id, city, state, postcode)
VALUES (00001, 'Sydney', 'NSW', 2050);

/* insert values into talbe MutualFund */
INSERT INTO MutualFund (symbol, name, c_date, t_num_shares, category,description)
VALUES ('TSLA', 'TESLA', '2011-01-01', 10000,'stocks', 'car');

/* insert values into talbe closing_price */
INSERT INTO closing_price (p_date, price, symbol)
VALUES ('2019-04-30', 30, 'TSLA');
INSERT INTO closing_price (p_date, price, symbol)
VALUES ('2019-04-29', 29, 'TSLA');
INSERT INTO closing_price (p_date, price, symbol)
VALUES ('2019-04-28', 28, 'TSLA');
INSERT INTO closing_price (p_date, price, symbol)
VALUES ('2019-04-27', 27, 'TSLA');

/* insert values into talbe "User" */
INSERT INTO "User" (login, password, address, email, name)
VALUES ('smallsixsnake', 'smart', 'Sydney', 'vicky@snake.com', 'Vicky');
INSERT INTO "User" (login, password, address, email, name)
VALUES ('Han', 'Strong', 'Sydney', 'WUHAN@HAN.com', 'WUHAN');
INSERT INTO "User" (login, password, address, email, name)
VALUES ('Bear', 'supersmart', 'Sydney', 'derek@smart.com', 'Derek');

/* insert values into talbe Customer */
INSERT INTO Customer (login, balance)
VALUES ('smallsixsnake', 2050.0);
INSERT INTO Customer (login, balance)
VALUES ('Han', 20000.0);

/* insert values into talbe Administrator */
INSERT INTO Administrator (login)
VALUES ('Bear');

/* insert values into talbe Transaction */
INSERT INTO Transaction (trans_id, sub_type)
VALUES (11111,'Deposit_Transaction');
INSERT INTO Transaction (trans_id, sub_type)
VALUES (11112,'Deposit_Transaction');
INSERT INTO Transaction (trans_id, sub_type)
VALUES (11113,'Deposit_Transaction');
INSERT INTO Transaction (trans_id, sub_type)
VALUES (21111,'Investment_Transaction');

/* insert values into talbe Deposit_Transaction */
INSERT INTO  Deposit_Transaction(trans_id, sub_type)
VALUES (11111,'Deposit_Transaction');
INSERT INTO  Deposit_Transaction(trans_id, sub_type)
VALUES (11112,'Deposit_Transaction');
INSERT INTO  Deposit_Transaction(trans_id, sub_type)
VALUES (11113,'Deposit_Transaction');

/* insert values into talbe Investment_Transaction */
INSERT INTO Investment_Transaction (trans_id, sub_type)
VALUES (21111,'Investment_Transaction');

/* insert values into talbe manages */
INSERT INTO manages (symbol, company_id)
VALUES ('TSLA',00001);

/* insert values into talbe deposit_trans */
INSERT INTO deposit_trans (trans_id, login, t_date, amount)
VALUES (11111,'smallsixsnake','2019-4-30',150.0);
INSERT INTO deposit_trans (trans_id, login, t_date, amount)
VALUES (11112,'Han','2019-4-30',300.0);

/* insert values into talbe prefers */
INSERT INTO prefers (login, symbol, percentage)
VALUES ('smallsixsnake','TSLA',0.5);
INSERT INTO prefers (login, symbol, percentage)
VALUES ('Han','TSLA',0.55);

/* insert values into talbe owns */
INSERT INTO owns (login, symbol, shares)
VALUES ('smallsixsnake','TSLA',20);

/* insert values into talbe portfolio_action */
INSERT INTO portfolio_action (trans_id, login, symbol, action, t_date, amount, price, num_shares)
VALUES (21111,'smallsixsnake', 'TSLA','buy', '2019-4-30', 150, 30, 5);






