# Informal CEO Message Draft

Hi Sudhir,

I wanted to share one important automation idea for our supplier data cleanup work.

Right now, when we get large supplier files from clients, the team manually sorts the same file multiple times by name, address, tax/VAT, domain/email, city/country, and secondary name fields to identify supplier clusters. This is very time-consuming, especially when the same supplier appears in different parts of a 50k or 100k row file.

I have been testing a supplier clustering engine concept using real historical client files. The goal is simple: upload a client supplier file and get the same file back with only two added columns: Cluster Number and Match Percentage. The tool should bring likely related supplier rows next to each other so the team reviews clusters instead of manually searching row by row.

This is not meant to fully replace human review from day one. The first goal is to automate the first pass and reduce manual sorting effort.

I already have a reference implementation package ready, plus test outputs from sample files. It includes name, address, tax/VAT, domain, secondary-name, and parent/family matching logic. We also found real edge cases like German address normalization, multiple tax fields, VAT inside JSON metadata, and parent/family bridge cases.

Can I get 2-3 people assigned to help take this forward properly? I would need engineering help to review the code structure, improve the logic, test with real files, and decide how we can deploy it internally.

I can own the business rules, testing, and validation with the enrichment team. I need engineering support to make it reliable and usable.
