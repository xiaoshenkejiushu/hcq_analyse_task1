# -*- coding: utf-8 -*-

import re

string = "500:20"
pattern = re.compile(r'\d+')
 
res = re.findall(pattern, string)
dis_rate = int(res[1])/int(res[0])
print(dis_rate)


str1 = 'fixed'
str1 = str1.replace('fixed','0')
print(str1)

