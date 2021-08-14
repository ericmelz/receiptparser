#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import json
import os
from decimal import *

MAX_PRODUCT_LEN=20


# In[2]:


# See https://stackoverflow.com/questions/1960516/python-json-serialize-a-decimal-object
class fakefloat(float):
    def __init__(self, value):
        self._value = value
    def __repr__(self):
        return str(self._value)
    
def defaultencode(o):
    if isinstance(o, Decimal):
        # Subclass float with custom repr?
        return fakefloat(o)
    raise TypeError(repr(o) + " is not JSON serializable")    


# In[102]:


class Address:
    street = None
    city = None
    state = None
    zipcode = None
    
    def __init__(self, street, city, state, zipcode):
        self.street = street
        self.city = city
        self.state = state
        self.zipcode = zipcode
    
    def __str__(self):
        return f'{self.street} {self.city} {self.state} {self.zipcode}'
    
    
class Header:
    address = None
    cashier = None
    phone = None
    receipt_date = None
    receipt_seqnum = None
    
    def __init__(self, address, cashier, phone, receipt_date, receipt_seqnum):
        self.address = address
        self.cashier = cashier
        self.phone = phone
        self.receipt_date = receipt_date
        self.receipt_seqnum = receipt_seqnum
        
    def __str__(self):
        return f'[address: {self.address}, cashier: {self.cashier}, phone: {self.phone}, \n ' +                f'receipt_date: {self.receipt_date}, receipt_seqnum: {self.receipt_seqnum}]'
    
    def to_dict(self):
        return {
                'address': self.address.__dict__ if self.address else None,
                'cashier': self.cashier,
                'phone': self.phone,
                'receipt_date': self.receipt_date,
                'receipt_seqnum': self.receipt_seqnum
               }


class LineItem:
    product = None
    qty = None
    unitprice = None
    total = None
    savings = None
    
    def __str__(self):
        return f'LineItem(product={self.product}, qty={self.qty}, unitprice={self.unitprice}, total={self.total}, ' +             f'savings={self.savings})'
    
    def to_dict(self):
        return self.__dict__
    
    
class Body:
    line_items = None
    tax = None
    savings = None
    balance = None
    datetime = None
    change = None
    
    def calc_balance(self):
        sum = Decimal('0')
        for item in self.line_items:
            if item.total:
                sum = sum + item.total 
        if self.tax:
            sum += self.tax
        return sum
            
    def calc_savings(self):
        sum = Decimal('0')
        for item in self.line_items:
            if item.savings:
                sum = sum + item.savings
            if item.total is not None and item.total < 0:
                if item.product != 'COUPON':
                    # TODO I think coupons that are brought by the user are not counted towards savings
                    # The other case that we're renaming as COUPON should be handled separately
                    sum = sum + item.total * -1
        return sum
    
    def calc_coupons(self):
        sum = Decimal('0')
        for item in self.line_items:
            if item.product == 'COUPON':
                sum = sum + item.total
        return sum
    
    def __init__(self, line_items, tax, savings, balance, datetime, change):
        self.line_items = line_items
        if tax is not None:
            self.tax = Decimal(str(tax))
        if savings is not None:
            self.savings = Decimal(str(savings))
        if balance is not None:
            self.balance = Decimal(str(balance))
        self.datetime = datetime
        if change is not None:
            self.change = Decimal(str(change))
        
    def __str__(self):
        return '----\n' +             f'tax: {self.tax}\n' +             f'savings: {self.savings}\n' +             f'calculated_savings: {self.calc_savings()}\n' +             f'balance: {self.balance}\n' +             f'calculated_balance: {self.calc_balance()}\n' +             f'coupon_total: {self.calc_coupons()}\n' +             f'datetime: {self.datetime}\n' +             f'change: {self.change}\n' +             '\n'.join([str(x) for x in self.line_items])
    
    def to_dict(self):
        return {
            'line_items': [item.to_dict() for item in self.line_items],
            'tax': self.tax,
            'savings': self.savings,
            'calculated_savings': self.calc_savings(),
            'balance': self.balance,
            'calculated_balance': self.calc_balance(),
            'coupon_total': self.calc_coupons(),
            'change': self.change,
            'datetime': self.datetime
        }

    
class Receipt:
    header = None
    Body = None
    
    def __init__(self, header, body):
        self.header = header
        self.body = body
        
    def __str__(self):
        return str(self.header) + '\n' + str(self.body)    
    
    def to_json(self):
        return json.dumps({
            'header': self.header.to_dict(),
            'body': self.body.to_dict()
        },
        indent=4,
        default=defaultencode)


## Tokens    
class ItemsAndQuantity:
    qty = None
    unitprice = None
    
    def __init__(self, qty, unitprice):
        self.qty = float(qty)
        self.unitprice = Decimal(unitprice)
    
    def __str__(self):
        return f'ItemsAndQuantity({self.qty}, {self.unitprice})'

    
class Product:
    name = None
    
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return f'Product({self.name})'

    
class ItemTotal:
    amount = None
    
    def __init__(self, amount):
        self.amount = Decimal(amount)
    
    def __str__(self):
        return f'ItemTotal({self.amount})'

    def to_json(self):
        return json.dumps(self.__dict__)

    
class Savings:
    amount = None
    
    def __init__(self, amount):
        self.amount = Decimal(amount)
    
    def __str__(self):
        return f'Savings({self.amount})'
    
    def to_json(self):
        return json.dumps(self.__dict__)
    

class Coupon:
    amount = None
    
    def __init__(self, amount):
        self.amount = Decimal(amount)
    
    def __str__(self):
        return f'Coupon({self.amount})'
    
    def to_json(self):
        return json.dumps(self.__dict__)
    

class MegaEvent:
    amount = None
    
    def __init__(self, amount):
        self.amount = Decimal(amount)
    
    def __str__(self):
        return f'MegaEvent({self.amount})'
    
    def to_json(self):
        return json.dumps(self.__dict__)
    

class GreenBagPts:
    amount = None
    
    def __init__(self, amount):
        self.amount = Decimal(amount)
    
    def __str__(self):
        return f'GreenBagPts({self.amount})'
    
    def to_json(self):
        return json.dumps(self.__dict__)
    

class Noise:
    noise = None
    
    def __init__(self, noise):
        self.noise = noise
        
    def __str__(self):
        return f'Noise({self.noise})'


# In[167]:


def parse_items_and_quantity(text):
    #'1.0 @ 10/10.00'
    m = re.match('(\d+\.\d+) @ (\d+)/(\d+\.\d+)', text)
    if m:
        qty = float(m[1])
        unitprice = float(m[3]) / float(m[2])
        return ItemsAndQuantity(qty, unitprice)
    #'2.0 @ 0.1'
    m = re.match('(\d+) @ (\d*\.\d+})', text)
    if m:
        return ItemsAndQuantity(m[1], m[2])
    m = re.match('(\d+\.\d+) @ (\d*\.\d+)', text)
    if m:
        return ItemsAndQuantity(m[1], m[2])
    m = re.match('(\d+\.\d+) lb @ (\d+\.\d+) /lb', text)
    if m:
        return ItemsAndQuantity(m[1], m[2])
    return None

def parse_item_total(text):
    m = re.match('(\d+\.\d+)-.', text)
    if m:
        d = Decimal(m[1])
        d = d * -1
        return ItemTotal(str(d))
    m = re.match('(\d+\.\d+)', text)
    if m:
        return ItemTotal(m[1])
    return None

def parse_savings_header(text):
    m = re.match('RALPHS SAVED YOU', text)
    if m:
        return True   
    m = re.match('.* (Sale)', text)
    if m:
        return True   
    m = re.match('.*%\s+off', text)
    if m:
        return True   
    
    return None

def parse_savings_body(text):
    m = re.match('(\d+\.\d+)', text)
    if m:
        return Savings(m[1])
    return None

def parse_coupon_header(text):
    m = re.match('SCANNED COUPON', text)
    if m:
        return True  
    return None

def parse_coupon_body(text):
    m = re.match('(\d+\.\d+)', text)
    if m:
        return Coupon(m[1])
    return None

def parse_megaevent_header(text):
    m = re.match('Mega Event Savings', text)
    if m:
        return True
    return None

def parse_megaevent_body(text):
    m = re.match('(\d+\.\d+)', text)
    if m:
        return MegaEvent(m[1])
    return None

def parse_greenbagpts_header(text):
    m = re.match('Green\s+Bag\s+Pts', text)
    if m:
        return True        
    return None

def parse_greenbagpts_body(text):
    m = re.match('(\d+)', text)
    if m:
        return GreenBagPts(m[1])
    return None

def parse_product(text):
    if len(text) == MAX_PRODUCT_LEN and text.endswith('RC'):
        text = text[:-2].strip()
    return Product(text)

def parse_noise(text):
    m = re.match('(^SC$|^RC$|^WT$|^MC$|^NP$|^MR$|^DB$|^<\+$|^.$)', text)
    if m:
        return Noise(m[1])
    m = re.match('(^AGE$|^VERIFICATION BYPASSED$)', text)
    if m:
        return Noise(m[1])
    return None

def maybe_merge(text1, text2, text3):
    '''
    Return <increment>, <text>
    where <increment> is the number of positions to advance, and <text> is the either <text1> or the merged text
    '''
    if text2 and text3:
        # Example(from 2020_04_19_01): '1.0', '@', '10/10.00'
        m1 = re.match('(\d+\.\d+)', text1)
        m2 = re.match('@', text2)
        m3 = re.match('\d+/\d+\.\d+', text3)
        if m1 and m2 and m3:
            return 3, f'{text1} {text2} {text3}'
        # Example(from 2020_04_19_01): '2', '@', '0.99'
        m1 = re.match('\d+', text1)
        m2 = re.match('@', text2)
        m3 = re.match('\d+\.\d+', text3)
        if m1 and m2 and m3:
            return 3, f'{text1} {text2} {text3}'
        
    if text2:
        m1 = re.match('(\d\.\d+)', text1)
        m2 = re.match('lb @ (\d\.\d+) /lb', text2)
        if m1 and m2:
            return 2, f'{text1} {text2}'
    return 1, text1

def merge_text(texts):    
    '''
    In some cases, text might get split.  For example the item and quanity "0.82 lb @ 0.99 /lb"
    can get split into "0.82" and "lb @ 0.99 /lb".
    If we detect those patterns, merge them
    '''
    i = 0
    result = []
    merged = False
    while i < len(texts):
        text1 = texts[i]
        text2 = texts[i+1] if i+1 < len(texts) else None
        text3 = texts[i+2] if i+2 < len(texts) else None
        increment, text = maybe_merge(text1, text2, text3)
        result.append(text)
        i += increment
    return result        

def parse_text(texts):
    '''
    Given a list of texts (strings) produce a list of tokens
    '''
    debug = False
    tokens = []
    i = -1
    while i < len(texts) - 1:
        i += 1
        text = texts[i]
        if debug:
            print(f'parsing |{text}|')
        token = parse_items_and_quantity(text)        
        if token:
            if debug:
                print(token)
            tokens.append(token)
            continue
        token = parse_noise(text)
        if token:
            if debug:
                print(token)
            continue
        token = parse_savings_header(text)
        if token:
            if debug:
                print(token)
            i += 1
            text = texts[i]
            token = parse_savings_body(text)
            if token:
                tokens.append(token)
                continue
        token = parse_coupon_header(text)
        if token:
            if debug:
                print(token)
            i += 1
            text = texts[i]
            token = parse_coupon_body(text)
            if token:
                if debug:
                    print(token)
                tokens.append(token)
                continue
        token = parse_megaevent_header(text)
        if token:
            if debug:
                print(token)
            i += 1
            text = texts[i]
            token = parse_megaevent_body(text)
            if token:
                if debug:
                    print(token)
                tokens.append(token)
                continue
        token = parse_greenbagpts_header(text)
        if token:
            if debug:
                print(token)
            i += 1
            text = texts[i]
            token = parse_greenbagpts_body(text)
            if token:
                if debug:
                    print(token)
                tokens.append(token)
                continue
        token = parse_item_total(text)
        if token:
            if debug:
                print(token)
            tokens.append(token)
            continue
        token = parse_product(text)
        if debug:
            print(token)
        tokens.append(token)
    return tokens
        

def parse_tokens(tokens):
    '''
    Given a list of tokens produce a list of line items
    '''
    items = []
    i = -1
    item = None
    while i < len(tokens) - 1:
        i += 1
        token = tokens[i]
        if isinstance(token, ItemsAndQuantity):
            item = LineItem()
            item.qty = token.qty
            item.unitprice = token.unitprice
            items.append(item)
            continue
        if isinstance(token, Product):
            if not item or item.total is not None:
                item = LineItem()
                items.append(item)
            item.product = token.name
            continue
        if isinstance(token, ItemTotal):
            item.total = token.amount
            continue
        if isinstance(token, Savings):
            tempitem = item
            if item.product == 'CA REDEM VAL':
                tempitem = items[-2]
            current_savings = tempitem.savings if tempitem.savings else 0
            tempitem.savings = token.amount + current_savings
        if isinstance(token, Coupon):
            coupon = LineItem()
            coupon.product = 'COUPON'
            coupon.total = token.amount * -1
            items.append(coupon)
        if isinstance(token, MegaEvent):
            mega = LineItem()
            mega.product = 'MEGAEVENT'
            mega.total = token.amount * -1
            items.append(mega)
    return items

def filter_line_items(items):
    return [i for i in items if not i.product.startswith('Trip Stakes')]
        
def find_and_extract(pattern, df):
    '''
    Find a pattern in df and extract the matching group.
    Return the extracting and the (row, col) location. 
    If not found or not matched, return None
    '''
    for col in range(df.shape[1]):
        df2 = df[df.columns[col]]
        if df2.dtype != np.dtype('O'):
            continue
        df2 = df[df.columns[col]].str.findall(pattern).str.len() > 0
        if not df2.any():
            continue
        row = df[df[df.columns[col]].str.findall(pattern).str.len() > 0].index[0]
        contents = df.iloc[row, col]
        m = re.match(pattern, contents)
        if not m:
            return None
        if m.lastindex < 1:
            return None
        extracted = []
        for i in range(1, m.lastindex + 1):
            extracted.append(m[i])
        return extracted, (row, col)
    return None

def extract_address(df):
    '''
    Return an address object
    '''
    city_state_zip_pattern = '(.*) (..) (\d{5})'
    street_pattern = '(\d+ .* (Blvd|St|Way))'
    street = None
    city = None
    state = None
    zipcode = None
    address = None
    res = find_and_extract(street_pattern, df)
    if res:
        street = res[0][0]
    res = find_and_extract(city_state_zip_pattern, df)
    if res:
        city_state_zip = res[0]
        city = city_state_zip[0]
        state = city_state_zip[1]
        zipcode = city_state_zip[2]
    if street or city or state or zipcode:
        address = Address(street, city, state, zipcode)
    return address
    
def extract_phone(df):
    phone = None
    res = find_and_extract('(\(\d{3}\) \d{3}-\d{4})', df)
    if res:
        phone = res[0][0]
    return phone

def extract_cashier(df):
    res = find_and_extract('Your cashier was (.*)', df)
    rowcol = (0, 0)
    cashier = None
    if res:
        cashier = res[0][0]
        rowcol = res[1]    
    return cashier, rowcol

def extract_header(df, receipt_date, receipt_seqnum):
    address = extract_address(df)
    cashier, _ = extract_cashier(df)
    phone = extract_phone(df)
    header = Header(address, cashier, phone, receipt_date, receipt_seqnum)
    return header
    

def find_rewards_customer_line(df):
    res = find_and_extract('(RALPHS rewards CUSTOMER)', df)
    if res:
        line, rowcol = res
        return rowcol[0]
    res = find_and_extract('(rewards CUSTOMER)', df)
    if res:
        line, rowcol = res
        return rowcol[0]
    raise Exception('Cant find rewards customer line')


def extract_line_item_text(df, start, end):
    texts = []
    for r in range(start, end):
        for c in range(df.shape[1]):
            text = df.iloc[r][c]
            if text and str(text) != 'nan':
                texts.append(str(text))
    return texts

def extract_line_items(df):
    _, (row, _) = extract_cashier(df)
    lineitem_start = row + 1
    lineitem_end = find_rewards_customer_line(df)
    texts = extract_line_item_text(df, lineitem_start, lineitem_end)
    texts = merge_text(texts)
    tokens = parse_text(texts)
    items = parse_tokens(tokens)
    return items

def find_nonnull_column(df, row, start_col):
    width = df.shape[1]
    for col in range(start_col, width):
        if not df.iloc[row:row+1, col:col+1].isnull().iloc[0,0]:
            return col  
        
def extract_tax(df):
    tax_res = find_and_extract('(TAX)',df)
    if tax_res:
        line, (row, col) = tax_res
        tax = df.iloc[row][find_nonnull_column(df, row, col+1)]
        return tax
    return None
    
def extract_savings(df):
    res = find_and_extract('(RALPHS rewards SAVINGS)',df)
    if res:
        line, (row, col) = res
        savings = df.iloc[row][find_nonnull_column(df, row, col+1)]
        return savings
    return None
    
def extract_balance(df):
    res = find_and_extract('.*(BALANCE)',df)
    if res:
        line, (row, col) = res
        balance = df.iloc[row][find_nonnull_column(df, row, col+1)]
        return balance
    return None

def extract_change(df):
    res = find_and_extract('(CHANGE)',df)
    if res:
        line, (row, col) = res
        change = df.iloc[row][find_nonnull_column(df, row, col+1)]
        return change
    return None

def extract_datetime(df):
    res = find_and_extract('(\d{2})/(\d{2})/(\d{2}) (\d{2}):(\d{2})(.m)', df)
    if res:
        parts = res[0]
        hour = parts[3]
        if parts[-1] == 'pm':
            hour = str(int(hour) + 12)
        
        datetime = f'20{parts[2]}-{parts[0]}-{parts[1]}T{hour}:{parts[4]}:00'
        return datetime
    return None
    
def extract_body(df):
    line_items = extract_line_items(df)
    line_items = filter_line_items(line_items)
    tax = extract_tax(df)
    savings = extract_savings(df)
    balance = extract_balance(df)
    datetime = extract_datetime(df)
    change = extract_change(df)
    body = Body(line_items, tax, savings, balance, datetime, change)
    return body

def validate_receipt(receipt):
    assert receipt.body.balance
    assert isinstance(receipt.body.balance, Decimal)
    assert receipt.body.balance > 0
    if receipt.body.savings:
        assert receipt.body.savings == receipt.body.calc_savings()
    assert receipt.body.balance == receipt.body.calc_balance()

def extract_receipt(df, receipt_date, receipt_seqnum):
    header = extract_header(df, receipt_date, receipt_seqnum)
    body = extract_body(df)
    receipt = Receipt(header, body)
    return receipt

def parse_receipt(excel_path, json_path, receipt_name, verbose=True, validate=True):
    receipt_excel_path = f'{excel_path}/{receipt_name}.xlsx'
    receipt_json_path = f'{json_path}/{receipt_name}.json'
    df = pd.read_excel(receipt_excel_path)
    receipt_date = receipt_name[:10]
    receipt_seqnum = receipt_name[11:]
    receipt = extract_receipt(df, receipt_date, receipt_seqnum)
    if validate:
        validate_receipt(receipt)
    if verbose:
        print(receipt)
    f = open(receipt_json_path, "w")
    j = receipt.to_json()
    if verbose:
        print(j)
    f.write(j)
    f.close()                  


# In[170]:


# For debugging

def debug():
    receipt_name = '2021_04_05_01'
    receipt_excel_path = f'{excel_path}/{receipt_name}.xlsx'
    df = pd.read_excel(receipt_excel_path)
    texts = extract_line_item_text(df, 4, find_rewards_customer_line(df))
    texts = merge_text(texts)
    print(texts)
    tokens = parse_text(texts)    
    for t in tokens:
        print(t)
    return df, tokens
        

#df,tokens=debug()


# In[169]:


chunk = 0
chunksize = 500
validate=True
#validate=False
problem_receipts = [
#    '2020_04_19_01',
                   ]

base_path = '/Users/ericmelz/Desktop/Ralphs/Receipts'
excel_path = f'{base_path}/Excel'
json_path = f'{base_path}/JSON'

l = os.listdir(excel_path)
all_receipt_names = sorted([f.split('.')[0] for f in l])

start_offset = chunksize * chunk
end_offset = start_offset + chunksize
receipt_names = all_receipt_names[start_offset:end_offset]

for receipt_name in receipt_names:
    print(f'Parsing {receipt_name}....', end='')
    if receipt_name in problem_receipts:
        print('SKIPPING!')
        continue
    print()
    parse_receipt(excel_path, json_path, receipt_name, verbose=False, validate=validate)


# In[ ]:




