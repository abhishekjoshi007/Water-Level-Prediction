from datetime import datetime

print("="*80)
print("STEP 2: DOWNLOADING CURRENT TCOON DATA")
print("="*80)

stations = {
    '005': 'Packery Channel',
    '008': 'Bob Hall Pier',
    '013': 'USS Lexington',
    '068': 'Port Aransas',
    '202': 'Corpus Christi Bay',
    '272': 'Aransas Pass',
    '015': 'Ingleside'
}

print("\n📥 MANUAL DOWNLOAD INSTRUCTIONS:")
print("-" * 80)
print("\n🌐 Go to: https://www.glo.texas.gov/coast/coastal-management/forms/tcoon-data.html")

print("\n📋 For each station below, download:")

for station_id, name in stations.items():
    print(f"\n{'='*60}")
    print(f"📍 STATION {station_id}: {name}")
    print(f"{'='*60}")
    
    print("\n   WATER LEVEL DATA:")
    print("   ┌─────────────────────────────────────────┐")
    print(f"   │ Station:        {station_id:25s} │")
    print("   │ Start Date:     2021-01-01              │")
    print(f"   │ End Date:       {datetime.now().strftime('%Y-%m-%d'):25s} │")
    print("   │ Parameter:      Water Level (pwl)       │")
    print("   │ Interval:       Hourly                  │")
    print("   │ Time Zone:      CST                     │")
    print("   │ Units:          Meters                  │")
    print("   │ Format:         Excel (.xls) or CSV     │")
    print("   └─────────────────────────────────────────┘")
    print(f"   💾 Save as: station_{station_id}_water_level_2021_2025.csv")
    
    print("\n   METEOROLOGICAL DATA:")
    print("   ┌─────────────────────────────────────────┐")
    print(f"   │ Station:        {station_id:25s} │")
    print("   │ Start Date:     2021-01-01              │")
    print(f"   │ End Date:       {datetime.now().strftime('%Y-%m-%d'):25s} │")
    print("   │ Parameters:     All Met (6 variables)   │")
    print("   │   - atp (Air Temperature)               │")
    print("   │   - wtp (Water Temperature)             │")
    print("   │   - wsd (Wind Speed)                    │")
    print("   │   - wgt (Wind Gust)                     │")
    print("   │   - wdr (Wind Direction)                │")
    print("   │   - bpr (Barometric Pressure)           │")
    print("   │ Interval:       Hourly                  │")
    print("   │ Format:         Excel (.xls) or CSV     │")
    print("   └─────────────────────────────────────────┘")
    print(f"   💾 Save as: station_{station_id}_met_2021_2025.csv")

print("\n" + "="*80)
print("📁 SAVE ALL FILES TO: data/fresh_download/")
print("="*80)
print("\n⏱️  Estimated time: 15-20 minutes for all stations")
print("💡 TIP: Download one station completely, then compare with your existing data")
print("         before downloading all stations")