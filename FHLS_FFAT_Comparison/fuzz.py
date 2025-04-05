# import json
# import logging

# # Load your converted vectors
# # with open("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/normal_rfuzz.json", "r") as f:
# # with open("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/overvoltage_rfuzz.json", "r") as f:
# # with open("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/clock_rfuzz.json", "r") as f:
# # with open("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/rowhammer_rfuzz.json", "r") as f:   
# with open("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/unknown_rfuzz.json", "r") as f:       
#     mutation_vectors = json.load(f)

# # Stub simulation function if you have no hardware RTL simulation
# def stub_simulation(vector):
#     # Simulate: arbitrarily count bits set to '1' as "coverage"
#     coverage = sum(vector)
#     # Simulate fault if high coverage arbitrarily
#     fault_detected = coverage > 30
#     return {"coverage": coverage, "fault": fault_detected}

# coverage_total = 0
# faults_total = 0

# logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# for idx, vector in enumerate(mutation_vectors):
#     result = stub_simulation(vector)
#     coverage_total += result["coverage"]
#     faults_total += 1 if result["fault"] else 0
#     # logging.info(f"Test input #{idx}: Coverage {result['coverage']}, Fault: {result['fault']}")

# logging.info(f"Total Inputs: {len(mutation_vectors)}")
# logging.info(f"Total Simulated Coverage: {coverage_total/len(mutation_vectors):.2f}")
# logging.info(f"Total Faults detected: {faults_total}")



import os
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# === Configuration ===
SEED_DIR = "seeds"
# Load your converted vectors
# with open("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/normal_rfuzz.json", "r") as f:
# with open("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/overvoltage_rfuzz.json", "r") as f:
# with open("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/clock_rfuzz.json", "r") as f:
# with open("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/rowhammer_rfuzz.json", "r") as f:   
# SEED_FILE = os.path.join(SEED_DIR,"E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HWF/unknown_data.hwf")    
# SEED_FILE = os.path.join(SEED_DIR,"E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HWF/normal_state.hwf") 
# SEED_FILE = os.path.join(SEED_DIR,"E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HWF/rowhammer.hwf") 
SEED_FILE = os.path.join(SEED_DIR,"E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HWF/overvoltage.hwf") 
# SEED_FILE = os.path.join(SEED_DIR,"E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HWF/clock_glitching.hwf")    
    # mutation_vectors = json.load(f)

# === Simulated Coverage & Fault Detection ===
def simulate_rfuzz_on_ffat(hwf_path):
    with open(hwf_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.startswith("CMD")]

    total_inputs = len(lines)
    coverage_score = 0.0
    fault_count = 0

    for line in lines:
        try:
            # Parse the line
            parts = line.split()
            voltage = float([p[1:] for p in parts if p.startswith("V")][0])
            duration = float([p[1:] for p in parts if p.startswith("D")][0])
            severity = float([p[1:] for p in parts if p.startswith("S")][0])

            # Simulated logic
            coverage_score += voltage * 0.2 + duration * 0.05
            if severity > 80 or voltage > 4.5 or duration > 10:
                fault_count += 1

        except Exception as e:
            logging.warning(f"Skipping malformed line: {line} | Error: {e}")
            continue

    return total_inputs, round(coverage_score, 2), fault_count

# === Main Evaluation ===
def main():
    logging.info(f"Evaluating FFAT-based input: {SEED_FILE}")
    if not os.path.exists(SEED_FILE):
        logging.error(f"Seed file not found: {SEED_FILE}")
        return

    start = time.time()
    total, coverage, faults = simulate_rfuzz_on_ffat(SEED_FILE)
    elapsed = time.time() - start

    print("\n========== FFAT x Rfuzz Simulation ==========")
    print(f"[INFO] Total Inputs            : {total}")
    print(f"[INFO] Total Simulated Coverage: {coverage}")
    print(f"[INFO] Total Faults Detected   : {faults}")
    print(f"[INFO] Elapsed Time            : {elapsed:.2f} sec")
    print("=============================================\n")

if __name__ == "__main__":
    main()