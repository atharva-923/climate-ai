# 🔧 ClimateAI India - Quick Troubleshooting Guide

## ✅ PRE-DEMO FINAL CHECKS

### 1. **Is the app running?**
```bash
# Check if app is accessible
curl http://localhost:8502
```
✅ **Your app is currently running on: http://localhost:8502**

### 2. **Quick App Test (2 minutes before demo)**

Open http://localhost:8502 and verify:

- [ ] ✅ Overview Dashboard loads with 4 metric cards
- [ ] ✅ 3D Temperature surface plot appears
- [ ] ✅ Extreme weather cards show data
- [ ] ✅ Location Analysis works (select Mumbai)
- [ ] ✅ Map shows all cities with markers
- [ ] ✅ Predictions work (select Bangalore)
- [ ] ✅ What-if Simulator sliders respond
- [ ] ✅ Model Performance shows metrics

---

## 🚨 COMMON ISSUES & FIXES

### Issue: App won't load / blank page
**Fix:**
```bash
cd climate-ai
streamlit run app/app.py
```

### Issue: Port already in use
**Fix:**
```bash
streamlit run app/app.py --server.port 8503
```
Then use: http://localhost:8503

### Issue: "Data not loaded" error
**Check:**
- Files exist: `data/processed/weather_processed.csv`
- Files exist: `data/processed/weather_featured.csv`
- Models exist: `models/*.pkl`

**Fix if missing:**
```bash
python src/preprocessing.py
python src/features.py
python src/train_temperature.py
python src/train_rainfall.py
```

### Issue: Charts not showing
**Fix:**
- Refresh the browser (Ctrl + F5)
- Clear Streamlit cache (click ⋮ menu → Clear cache)

### Issue: Slow performance
**Fix:**
- Close other heavy applications
- Restart the app
- Use a lighter browser (Chrome recommended)

---

## 🎬 LAST MINUTE DEMO PREP

### 30 Minutes Before:
1. ✅ Restart your computer (fresh start)
2. ✅ Close all unnecessary apps
3. ✅ Open only: Browser + Streamlit app
4. ✅ Test the full demo flow once
5. ✅ Have water ready 💧

### 10 Minutes Before:
1. ✅ Open http://localhost:8502
2. ✅ Pre-load Overview Dashboard
3. ✅ Close unnecessary browser tabs
4. ✅ Put phone on silent
5. ✅ Take a deep breath 😊

### 2 Minutes Before:
1. ✅ Refresh the app page
2. ✅ Check audio/video if virtual
3. ✅ Have DEMO_GUIDE.md open for reference
4. ✅ You've got this! 🚀

---

## 📊 CURRENT PROJECT STATUS

**✅ COMPLETE:**
- ✅ Data processed (182,600 records)
- ✅ Models trained (Temperature + Rainfall)
- ✅ 6 Dashboards built with 3D effects
- ✅ App running on port 8502
- ✅ All visualizations working
- ✅ Demo guide created

**📍 YOUR APP:**
- **URL:** http://localhost:8502
- **Status:** 🟢 Running
- **Ready:** YES! 🎉

---

## 💪 CONFIDENCE BOOSTERS

1. Your model accuracy is EXCELLENT (0.86°C MAE, 92.7% R²)
2. The 3D visualizations are STUNNING
3. The What-If Simulator is UNIQUE
4. You have 50 cities and 182K records
5. Everything is WORKING

**You are 100% ready to demo! 🌟**

---

## 🆘 EMERGENCY CONTACTS

If something breaks during demo:
1. Stay calm
2. Refresh the page
3. If still broken, explain: "This is the prediction accuracy we achieved: 0.86°C MAE"
4. Show the DEMO_GUIDE.md screenshots if app fails

**Backup plan:** Have the README.md open showing project stats

---

## ⏰ TIME CHECK
- Current time: ~12:53 PM IST
- App is running
- You're ready!

**GOOD LUCK! YOU'VE GOT THIS! 🚀🌟**
