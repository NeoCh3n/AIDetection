import requests
import csv
import os

# ======== configuration =========
QRADAR_HOST = "https://<qradar_host>"   # Replace it with your QRadar console address
API_TOKEN = "<api_token>"          # Generate it in Admin → Authorized Services
API_VERSION = "20.0"                    # Adjust according to your QRadar version (optional 12.0~17.0)
VERIFY_SSL = False                      # Set to False if the certificate is not installed correctly

# ======== API endpoints =========
API_ENDPOINTS = {
    "rules": "/api/analytics/rules",
    "ade_rules": "/api/analytics/ade_rules", 
    "building_blocks": "/api/analytics/building_blocks"
}

# ======== Output configuration =========
OUTPUT_DIR = "Qradar_rule"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILES = {
    "rules": os.path.join(OUTPUT_DIR, "qradar_rules.csv"),
    "ade_rules": os.path.join(OUTPUT_DIR, "qradar_aderules.csv"),
    "building_blocks": os.path.join(OUTPUT_DIR, "qradar_buildingblocks.csv")
}

# ======== API requests =========
headers = {
    "SEC": API_TOKEN,
    "Version": API_VERSION
}

def fetch_and_save_rules(endpoint_name, endpoint_path, output_file):
    """Fetch rules from a specific endpoint and save to CSV"""
    url = f"{QRADAR_HOST}{endpoint_path}"
    
    try:
        response = requests.get(url, headers=headers, verify=VERIFY_SSL)
        response.raise_for_status()
        rules = response.json()
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "type", "enabled", "origin"])
            
            for r in rules:
                # Handle both rule and building block formats
                rule_data = {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "type": r.get("type", r.get("rule_type", "")),
                    "enabled": r.get("enabled"),
                    "origin": r.get("origin", "")
                }
                writer.writerow([
                    rule_data["id"],
                    rule_data["name"],
                    rule_data["type"],
                    rule_data["enabled"],
                    rule_data["origin"]
                ])
        
        print(f"[✔] Exported {len(rules)} {endpoint_name} to {output_file}")
        return len(rules)
        
    except requests.exceptions.RequestException as e:
        print(f"[✘] Failed to fetch {endpoint_name}: {e}")
        return 0
    except Exception as e:
        print(f"[✘] Error processing {endpoint_name}: {e}")
        return 0

# ======== Fetch all rule types =========
total_rules = 0
for rule_type, endpoint in API_ENDPOINTS.items():
    count = fetch_and_save_rules(rule_type, endpoint, OUTPUT_FILES[rule_type])
    total_rules += count

print(f"[✔] Total rules exported: {total_rules}")