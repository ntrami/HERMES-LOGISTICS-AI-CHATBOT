import csv
import random
from datetime import datetime, timedelta

routes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
warehouses = ['WH1', 'WH2', 'WH3', 'WH4']
reasons = ['Weather', 'Traffic', 'Mechanical', 'None']

start = datetime(2024, 1, 1)
# Use current date (10.12.2025) as end date instead of fixed date
end = datetime(2025, 12, 10)  # Current date: December 10, 2025
total_days = (end - start).days

rows = []
for i in range(1, 5001):
    delta_days = random.randint(0, total_days)
    dt = (start + timedelta(days=delta_days)).date()
    rows.append({
        'id': i,
        'route': random.choice(routes),
        'warehouse': random.choice(warehouses),
        'delivery_time': random.randint(1, 10),
        'delay_minutes': random.randint(0, 120),
        'delay_reason': random.choice(reasons),
        'date': dt.isoformat()
    })

import os
os.makedirs('data', exist_ok=True)

with open('data/shipments.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'route', 'warehouse', 'delivery_time', 
                                           'delay_minutes', 'delay_reason', 'date'])
    writer.writeheader()
    writer.writerows(rows)

print(f'Generated {len(rows)} rows in data/shipments.csv')


