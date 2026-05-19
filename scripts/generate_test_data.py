#!/usr/bin/env python3
"""Generate realistic test supplier data for testing the clustering engine."""

import random
import csv
import os

random.seed(42)

# Sample data for generating realistic suppliers
COMPANY_NAMES = [
    ("ABS Safety GmbH", "DE", "Gewerbering 3", "Kevelaer", "DE233002380", "abs-safety.de"),
    ("GreenPharma S.A.", "FR", "3 Allée du Titane", "Orléans", "FR12345678901", "greenpharma.fr"),
    ("Altis Recruitment & Technology Inc.", "CA", "100 King St W", "Toronto", "CA123456789", "altis.com"),
    ("427 QEW Kia", "CA", "427/QEW Highway", "Mississauga", "", "427kia.ca"),
    ("H.Y. Louie", "CA", "123 Main St", "Vancouver", "", "hylouie.com"),
    ("A & B One Kommunikationsagentur GmbH", "DE", "Frankfurter Str 10", "Frankfurt", "", "abone.de"),
    ("C and T Investments Inc", "US", "500 W Madison St", "Chicago", "06-1594084", "ctinvestments.com"),
    ("ICEE CO", "US", "1000 Corporate Ave", "Los Angeles", "95-2499371", "icee.com"),
    ("CCBC Northern New England", "US", "1 Coca Cola Dr", "Bedford", "04-2614952", "ccbcne.com"),
    ("Merelex Corporation", "US", "10884 Weyburn Ave", "Los Angeles", "", "americanelements.com"),
    ("American Elements", "US", "10884 Weyburn Ave", "Los Angeles", "", "americanelements.com"),
    ("ADNKRONOS SPA", "IT", "Via della Magliana 900", "Roma", "IT12345678901", "adnkronos.com"),
    ("ADNKRONOS Salute SRL", "IT", "Via della Magliana 900", "Roma", "IT98765432109", "adnkronos.com"),
    ("Indeed Inc", "US", "6433 Champion Grandview Way", "Austin", "", "indeed.com"),
    ("Indeed Brasil Pesquisa de Empregos", "BR", "Av. Pres. Juscelino Kubitschek", "São Paulo", "", "indeed.com"),
    ("Thermo Electron GmbH", "DE", "Haberstrasse 10", "Dreieich", "", "thermofisher.com"),
    ("Thermo Electron SAS", "FR", "3 rue Léon Blum", "Courtaboeuf", "", "thermofisher.com"),
    ("Viatris APS", "DK", "Dampskibsselskabsvej 30", "København", "", "viatris.com"),
    ("MEDA AS", "DK", "Dampskibsselskabsvej 30", "København", "", "viatris.com"),
    ("Subway", "US", "325 Bic Dr", "Milford", "", "subway.com"),
    ("Capitol Subway 25542", "US", "325 Bic Dr", "Milford", "", "subway.com"),
    ("Hillsdale Subway 22818", "US", "325 Bic Dr", "Milford", "", "subway.com"),
    ("Shoppers Drug Mart #313", "CA", "243 Consumers Rd", "Toronto", "", "shoppersdrugmart.ca"),
    ("Shoppers Drug Mart #384", "CA", "5000 Yonge St", "Toronto", "", "shoppersdrugmart.ca"),
    ("Apple Auto Glass", "CA", "123 Industrial Rd", "Langley", "", "appleautoglass.ca"),
    ("Apple Auto Glass Langley", "CA", "456 Fraser Hwy", "Langley", "", "appleautoglass.ca"),
    ("Apple Auto Glass Surrey", "CA", "789 King George Blvd", "Surrey", "", "appleautoglass.ca"),
    ("Battleford Esso", "CA", "100 Highway 40", "Battleford", "", "esso.ca"),
    ("Biggar Esso", "CA", "200 Main St", "Biggar", "", "esso.ca"),
    ("DHL Express", "DE", "Charles-de-Gaulle-Str. 20", "Bonn", "", "dhl.com"),
    ("DHL Global Forwarding", "DE", "Charles-de-Gaulle-Str. 20", "Bonn", "", "dhl.com"),
    ("DHL Supply Chain", "US", "1801 International Dr", "Westerville", "", "dhl.com"),
    ("IQVIA", "US", "4820 Emperor Blvd", "Durham", "", "iqvia.com"),
    ("IQVIA Australia Pty Ltd", "AU", "Level 5, 456 St Kilda Rd", "Melbourne", "", "iqvia.com"),
    ("IQVIA France SAS", "FR", "17 avenue Georges Pompidou", "Lyon", "", "iqvia.com"),
    ("Danone", "FR", "17 boulevard Haussmann", "Paris", "", "danone.com"),
    ("Danone Deutschland GmbH", "DE", "Oskar-Jäger-Str. 107", "Köln", "", "danone.com"),
    ("Danone Mexico S.A. de C.V.", "MX", "Av. Insurgentes Sur 1605", "Mexico City", "", "danone.com"),
    ("RBC Royal Bank", "CA", "200 Bay St", "Toronto", "", "rbc.com"),
    ("Royal Bank of Canada", "CA", "200 Bay St", "Toronto", "", "rbc.com"),
    ("Royal Bank Visa", "CA", "200 Bay St", "Toronto", "", "rbc.com"),
    ("Pitney Bowes", "US", "3001 Summer St", "Stamford", "", "pitneybowes.com"),
    ("Pitney Bowes Leasing", "US", "3001 Summer St", "Stamford", "", "pitneybowes.com"),
    ("Pitney Bowes of Canada", "CA", "3001 Summer St", "Stamford", "", "pitneybowes.com"),
    ("CBRE", "US", "2100 McKinney Ave", "Dallas", "", "cbre.com"),
    ("CBRE GWS", "US", "2100 McKinney Ave", "Dallas", "", "cbre.com"),
    ("Alberta Health Services", "CA", "10004 104 Ave NW", "Edmonton", "", "albertahealthservices.ca"),
    ("AHS", "CA", "10004 104 Ave NW", "Edmonton", "", "albertahealthservices.ca"),
    ("Minister of Finance", "CA", "Frost Building South", "Toronto", "", "ontario.ca"),
    ("Ministry of Finance", "CA", "Frost Building South", "Toronto", "", "ontario.ca"),
    ("2230434 Ontario Inc", "CA", "123 Bay St", "Toronto", "123456789RC0001", ""),
    ("2230434 Ontario Ltd", "CA", "123 Bay St", "Toronto", "123456789RC0001", ""),
    ("Schlienkamp, Anja", "DE", "Musterstraße 1", "Berlin", "", "wallmeyer.de"),
    ("Wallmeyer GmbH", "DE", "Musterstraße 1", "Berlin", "", "wallmeyer.de"),
    ("RMHC of Ohio Valley Inc", "US", "123 Main St", "Louisville", "", "rmhc.org"),
    ("Ronald McDonald House Charities of the Ohio Valley Inc", "US", "123 Main St", "Louisville", "", "rmhc.org"),
    ("AT and T Mobility Puerto Rico", "PR", "250 Plaza Suite 100", "San Juan", "", "att.com"),
    ("Liberty Mobile Puerto Rico Inc", "PR", "250 Plaza Suite 100", "San Juan", "", "att.com"),
    ("Eagle-Picher Industries", "US", "123 Industrial Blvd", "Joplin", "", "eaglepicher.com"),
    ("Eagle-Picher Technologies GmbH", "DE", "Industriestraße 5", "Düsseldorf", "", "eaglepicher.com"),
    ("ACMI Beverage SPA", "IT", "Via dell'Industria 10", "Bologna", "", "acmi.it"),
    ("ACMI Beverage Iberica SL", "ES", "Calle Mayor 50", "Madrid", "", "acmi.it"),
    ("ACMI Mexico SA de CV", "MX", "Av. Reforma 100", "Mexico City", "", "acmi.it"),
    ("TGS Baltic Lithuania", "LT", "Konstitucijos pr. 7", "Vilnius", "", "tgsbaltic.lt"),
    ("TGS Baltic Estonia", "EE", "Narva mnt 5", "Tallinn", "", "tgsbaltic.lt"),
    ("TGS Baltic Latvia", "LV", "Brivibas iela 40", "Riga", "", "tgsbaltic.lt"),
    ("ICON plc", "IE", "South County Business Park", "Dublin", "", "iconplc.com"),
    ("PRA Health Sciences", "US", "4130 Parklake Ave", "Raleigh", "", "iconplc.com"),
    ("Convergint Technologies", "US", "855 W Victoria St", "Schaumburg", "", "convergint.com"),
    ("ICD Security", "US", "855 W Victoria St", "Schaumburg", "", "convergint.com"),
    ("Bell Canada", "CA", "1 Carrefour Alexander-Graham-Bell", "Verdun", "", "bell.ca"),
    ("Bell Aliant", "CA", "1 Carrefour Alexander-Graham-Bell", "Verdun", "", "bell.ca"),
    ("Telus", "CA", "25 York St", "Toronto", "", "telus.com"),
    ("Medisys", "CA", "25 York St", "Toronto", "", "telus.com"),
    ("Emergis", "CA", "25 York St", "Toronto", "", "telus.com"),
    ("Patheon", "NL", "Westerduinweg 2", "Amsterdam", "", "thermofisher.com"),
    ("Thermo Fisher Scientific", "US", "168 Third Ave", "Waltham", "", "thermofisher.com"),
    ("Sigma Aldrich", "US", "3050 Spruce St", "St. Louis", "", "merckgroup.com"),
    ("Millipore", "US", "80 Ashby Rd", "Burlington", "", "merckgroup.com"),
    ("Merck KGaA", "DE", "Frankfurter Str. 250", "Darmstadt", "", "merckgroup.com"),
    ("MEDA", "SE", "Box 304", "Solna", "", "viatris.com"),
    ("Viatris", "US", "1000 Mylan Blvd", "Canonsburg", "", "viatris.com"),
    ("AppExtremes LLC", "US", "123 Tech Blvd", "San Francisco", "", "conga.com"),
    ("Conga", "US", "123 Tech Blvd", "San Francisco", "", "conga.com"),
    ("C and T Investments Inc", "US", "500 W Madison St", "Chicago", "06-1594084", "ctinvestments.com"),
    ("DQ Grill and Chill", "US", "500 W Madison St", "Chicago", "", "dairyqueen.com"),
    ("Alexander Willowridge Industrial", "CA", "123 Willowridge Dr", "Calgary", "", "willowridgeconstruction.ca"),
    ("Willowridge Construction Ltd", "CA", "123 Willowridge Dr", "Calgary", "", "willowridgeconstruction.ca"),
    ("ABC-CZEPCZYŃSKI SP.ZO.O. SP.K.", "PL", "ul. Warszawska 1", "Warszawa", "", "abc-czepczynski.pl"),
    ("ABC - CZEPCZYŃSKI SP. Z O.O. SP. K.", "PL", "ul. Warszawska 1", "Warszawa", "", "abc-czepczynski.pl"),
    ("1-800-GOT-JUNK?", "CA", "1 Got Junk Way", "Vancouver", "", "1800gotjunk.com"),
    ("1-800-GOT-JUNK? (DEACTIVATED)", "CA", "1 Got Junk Way", "Vancouver", "", "1800gotjunk.com"),
    ("ABS-Beschichtungen GmbH", "DE", "Ludwig-Erhard-Str 10", "Kevelaer", "", "abs-beschichtungen.de"),
    ("ABSAtec GmbH", "DE", "Riedheimerstraße 8/1", "Mannheim", "", "absatec.de"),
    ("Gestión Boussias Fortin", "CA", "123 Rue Saint-Jean", "Québec", "", ""),
    ("8729794 Canada Inc", "CA", "123 Rue Saint-Jean", "Québec", "", ""),
    ("A+B One Kommunikationsagentur GmbH", "DE", "Frankfurter Str 10", "Frankfurt", "", "abone.de"),
    ("A & B One Kommunikationsagentur GmbH", "DE", "Frankfurter Str 10", "Frankfurt", "", "abone.de"),
    ("Canarama Shell", "CA", "100 Highway 16", "Edson", "", "shell.ca"),
    ("Canarama Shell #44163", "CA", "200 Main St", "Edson", "", "shell.ca"),
    ("Boylan Pharmasave #302", "CA", "123 Main St", "Toronto", "", "pharmasave.com"),
    ("Boylan Pharmasave #303", "CA", "456 Queen St", "Toronto", "", "pharmasave.com"),
    ("Blackfalds Shell & Foodstore", "CA", "100 Broadway Ave", "Blackfalds", "", "shell.ca"),
    ("Cochrane Shell", "CA", "200 Railway Ave", "Cochrane", "", "shell.ca"),
    ("IGA", "CA", "123 Grocery Ln", "Vancouver", "", "iga.ca"),
    ("IGA Supermarket", "CA", "456 Food St", "Calgary", "", "iga.ca"),
    ("Fas Gas", "CA", "100 Fuel Rd", "Edmonton", "", "fasgas.com"),
    ("Fas Gas Plus", "CA", "200 Petro Blvd", "Calgary", "", "fasgas.com"),
    ("Sobeys", "CA", "123 Grocery Way", "Stellarton", "", "sobeys.com"),
    ("Co-op Gas Bar", "CA", "100 Coop Dr", "Saskatoon", "", "coop.ca"),
    ("Canadian Tire", "CA", "1 Canadian Tire Rd", "Toronto", "", "canadiantire.ca"),
    ("Petro Canada", "CA", "100 Petro Way", "Calgary", "", "petro-canada.ca"),
    ("Shell", "CA", "1 Shell Plaza", "Calgary", "", "shell.ca"),
    ("Esso", "CA", "100 Esso Blvd", "Toronto", "", "esso.ca"),
    ("Popeyes", "US", "1 Popeyes Plaza", "Miami", "", "popeyes.com"),
    ("Dairy Queen", "US", "1 DQ Way", "Minneapolis", "", "dairyqueen.com"),
    ("Regents of the University of California", "US", "1111 Franklin St", "Oakland", "", "ucop.edu"),
    ("University of California Berkeley", "US", "200 California Hall", "Berkeley", "", "berkeley.edu"),
    ("University of California Los Angeles", "US", "405 Hilgard Ave", "Los Angeles", "", "ucla.edu"),
    ("Research Institute of McGill", "CA", "845 Sherbrooke St W", "Montreal", "", "mcgill.ca"),
    ("McGill University", "CA", "845 Sherbrooke St W", "Montreal", "", "mcgill.ca"),
    ("DSMZ", "DE", "Inhoffenstraße 7B", "Braunschweig", "", "dsmz.de"),
    ("Leibniz-Institut DSMZ", "DE", "Inhoffenstraße 7B", "Braunschweig", "", "dsmz.de"),
    ("ASCO", "US", "2318 Mill Rd", "Alexandria", "", "asco.org"),
    ("American Cancer Society", "US", "250 Williams St NW", "Atlanta", "", "cancer.org"),
    ("Fisher Clinical Services", "US", "1000 Fischer Dr", "Allentown", "", "thermofisher.com"),
    ("CBRE Excellerate", "US", "2100 McKinney Ave", "Dallas", "", "cbre.com"),
    ("Cascade Thermal", "US", "2100 McKinney Ave", "Dallas", "", "cbre.com"),
    ("Mercer", "US", "1166 Avenue of the Americas", "New York", "", "mercer.com"),
    ("Darwin", "US", "1166 Avenue of the Americas", "New York", "", "mercer.com"),
    ("MMC", "US", "1166 Avenue of the Americas", "New York", "", "mercer.com"),
    ("Kantar", "GB", "12 Hammersmith Grove", "London", "", "kantar.com"),
    ("Ogilvy", "US", "636 11th Ave", "New York", "", "ogilvy.com"),
    ("Ricoh", "JP", "8-13-1 Ginza", "Tokyo", "", "ricoh.com"),
    ("Jones Lang LaSalle", "US", "200 E Randolph St", "Chicago", "", "jll.com"),
    ("JLL", "US", "200 E Randolph St", "Chicago", "", "jll.com"),
    ("Eurofins", "LU", "23 Val Fleuri", "Luxembourg", "", "eurofins.com"),
    ("Nutricia", "NL", "Zonnebaan 30", "Utrecht", "", "danone.com"),
    ("GS1", "BE", "Avenue Louise 326", "Brussels", "", "gs1.org"),
    ("Iron Mountain", "US", "745 Atlantic Ave", "Boston", "", "ironmountain.com"),
    ("Dell", "US", "1 Dell Way", "Round Rock", "", "dell.com"),
    ("Dell Technologies", "US", "1 Dell Way", "Round Rock", "", "dell.com"),
    ("VMware", "US", "3401 Hillview Ave", "Palo Alto", "", "broadcom.com"),
]

# Generate variations for duplicates
def generate_variations(base_record, n_variations=2):
    """Generate realistic variations of a supplier record."""
    variations = [base_record]
    name, country, address, city, tax, domain = base_record

    for i in range(n_variations):
        variant_name = name
        variant_address = address
        variant_tax = tax

        # Case variation
        if random.random() < 0.3:
            variant_name = name.upper()

        # Punctuation variation
        if "&" in name and random.random() < 0.3:
            variant_name = name.replace("&", " and ")
        if "and" in name.lower() and random.random() < 0.3:
            variant_name = name.replace(" and ", " & ").replace(" AND ", " & ")

        # Legal suffix variation
        if "Inc" in name and random.random() < 0.2:
            variant_name = name.replace("Inc", "Incorporated")
        if "Ltd" in name and random.random() < 0.2:
            variant_name = name.replace("Ltd", "Limited")
        if "GmbH" in name and random.random() < 0.2:
            variant_name = name.replace("GmbH", "Gmbh")

        # Status label
        if random.random() < 0.1:
            variant_name = name + " (DEACTIVATED)"

        # Address abbreviation variation
        if "Street" in address and random.random() < 0.3:
            variant_address = address.replace("Street", "St").replace("street", "st")
        if "Avenue" in address and random.random() < 0.3:
            variant_address = address.replace("Avenue", "Ave")
        if "Road" in address and random.random() < 0.3:
            variant_address = address.replace("Road", "Rd")

        # Accent variation
        if "é" in name and random.random() < 0.3:
            variant_name = name.replace("é", "e")
        if "ü" in name and random.random() < 0.3:
            variant_name = name.replace("ü", "ue")
        if "ß" in name and random.random() < 0.3:
            variant_name = name.replace("ß", "ss")

        # Missing tax
        if tax and random.random() < 0.3:
            variant_tax = ""

        # Different address (same entity, multiple locations)
        if random.random() < 0.2:
            variant_address = f"{random.randint(100, 999)} {address.split(' ', 1)[1] if ' ' in address else address}"
            variant_city = random.choice(["Toronto", "Vancouver", "Calgary", "Montreal", "Ottawa"]) if country == "CA" else city
        else:
            variant_city = city

        variations.append((variant_name, country, variant_address, variant_city, variant_tax, domain))

    return variations


def generate_csv(filename, n_records=100):
    """Generate test CSV file with n_records rows."""

    # Select random subset of base companies
    selected = random.choices(COMPANY_NAMES, k=min(n_records // 2, len(COMPANY_NAMES)))

    records = []
    for company in selected:
        variations = generate_variations(company, n_variations=random.randint(1, 3))
        records.extend(variations)

    # Add some singletons (unique companies)
    singleton_names = [
        (f"Unique Supplier {i}", "US", f"{random.randint(100, 9999)} Main St", 
         random.choice(["New York", "Chicago", "Boston", "Seattle"]), 
         "", f"unique{i}.com")
        for i in range(max(0, n_records - len(records)))
    ]
    records.extend(singleton_names)

    # Shuffle
    random.shuffle(records)
    records = records[:n_records]

    # Write CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Supplier Name', 'Address', 'City', 'Country', 'Postal Code',
            'Tax ID', 'Email', 'Website', 'Name 2', 'Name 3'
        ])

        for name, country, address, city, tax, domain in records:
            postal = f"{random.randint(10000, 99999)}" if country == "US" else f"{random.randint(10000, 99999)}"
            email = f"contact@{domain}" if domain else f"user{random.randint(1,999)}@gmail.com"
            website = f"https://{domain}" if domain else ""
            name2 = ""
            name3 = ""

            # Add some DBA relationships in Name 2
            if "Subway" in name:
                name2 = "Doctor's Associates LLC"
            elif "DQ" in name or "Dairy Queen" in name:
                name2 = "International Dairy Queen Inc"
            elif "Popeyes" in name:
                name2 = "AFC Enterprises Inc"

            writer.writerow([name, address, city, country, postal, tax, email, website, name2, name3])

    print(f"Generated {filename} with {len(records)} rows")
    return len(records)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_csv("data/sample_suppliers_100.csv", 100)
    generate_csv("data/sample_suppliers_1000.csv", 1000)
