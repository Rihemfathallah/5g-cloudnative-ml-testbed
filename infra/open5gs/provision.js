// mongosh script: upsert the test subscriber(s) from subscriber.json.
// IMSI range: the base IMSI plus N-1 consecutive ones (UERANSIM profile registers several UEs).
const fs = require('fs');
const base = JSON.parse(fs.readFileSync('/provision/subscriber.json', 'utf8'));
const n = parseInt(process.env.N_SUBSCRIBERS || '1');
for (let i = 0; i < n; i++) {
  const doc = JSON.parse(JSON.stringify(base));
  doc.imsi = (BigInt(base.imsi) + BigInt(i)).toString().padStart(15, '0');
  doc.security.sqn = NumberLong(doc.security.sqn);
  db.getSiblingDB('open5gs').subscribers.updateOne({imsi: doc.imsi}, {$set: doc}, {upsert: true});
}
print('subscribers:', db.getSiblingDB('open5gs').subscribers.countDocuments());
