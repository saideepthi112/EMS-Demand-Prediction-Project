import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple

FACILITY_TYPES = [
    "Trauma-Related Incidents",
    "Environmental and Poisoning Emergencies",
    "Medical Emergencies",
    "Other"
]

class HospitalRanker:
    def __init__(self, hospitals: List[Dict[str, Any]], target_zip: str):
        self.target_zip = target_zip
        self.by_facility: Dict[str, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
        self.by_facility_expertise: Dict[str, Dict[str, List[Tuple[float, Dict[str, Any]]]]] = defaultdict(lambda: defaultdict(list))

        self._build_index(hospitals)
        self._finalize()

    def compute_score(self, hosp: Dict[str, Any]) -> float:
        quality = float(hosp.get("quality", 0))
        beds = float(hosp.get("beds", 0))
        doctors = float(hosp.get("doctors", 0))
        return quality + 0.1 * beds + 0.05 * doctors

    def _build_index(self, hospitals: List[Dict[str, Any]]):
        for hosp in hospitals:
            if str(hosp.get("zip")) != self.target_zip:
                continue

            facilities = hosp.get("facilities", [])
            expertises = hosp.get("expertise", [])
            if not isinstance(expertises, list):
                expertises = [expertises] if expertises else []

            score = self.compute_score(hosp)

            for facility in facilities:
                if facility in FACILITY_TYPES:
                    self.by_facility[facility].append((score, hosp))
                    for exp in expertises:
                        self.by_facility_expertise[facility][exp].append((score, hosp))

    def _finalize(self):
        for facility in self.by_facility:
            self.by_facility[facility].sort(key=lambda x: x[0], reverse=True)

        for facility in self.by_facility_expertise:
            for expertise in self.by_facility_expertise[facility]:
                self.by_facility_expertise[facility][expertise].sort(key=lambda x: x[0], reverse=True)

    def top_hospitals_by_facility(self, facility: str, top_n: int = 5) -> List[Dict[str, Any]]:
        return [h for _, h in self.by_facility.get(facility, [])[:top_n]]

    def top_hospitals_by_facility_and_expertise(self, facility: str, expertise: str, top_n: int = 5) -> List[Dict[str, Any]]:
        return [h for _, h in self.by_facility_expertise.get(facility, {}).get(expertise, [])[:top_n]]


if __name__ == "__main__":
    with open("deduplicated_hospitals_10002.json", "r") as f:
        hospitals = json.load(f)

    zip_code = "10002"
    ranker = HospitalRanker(hospitals, zip_code)

    # Top 3 Medical Emergencies hospitals
    print("Top 3 Medical Emergencies hospitals:")
    top_medical = ranker.top_hospitals_by_facility("Medical Emergencies", top_n=3)
    for h in top_medical:
        print(f" - {h['name']} (score: {ranker.compute_score(h):.2f}, expertise: {h.get('expertise')})")

    # Top 3 Medical Emergencies hospitals with INTERNAL MEDICINE expertise
    print("\nTop 3 Medical Emergencies hospitals with INTERNAL MEDICINE expertise:")
    top_internal = ranker.top_hospitals_by_facility_and_expertise("Medical Emergencies", "INTERNAL MEDICINE", top_n=3)
    for h in top_internal:
        print(f" - {h['name']} (score: {ranker.compute_score(h):.2f})")
