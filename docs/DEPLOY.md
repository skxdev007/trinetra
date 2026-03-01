# 🚀 Quick Deploy Guide

## Deploy to GitHub Pages in 3 Steps

### Step 1: Commit and Push
```bash
git add docs/
git commit -m "Add SHARINGAN-DEEP documentation website"
git push origin main
```

### Step 2: Enable GitHub Pages
1. Go to: `https://github.com/YOUR_USERNAME/YOUR_REPO/settings/pages`
2. Under **Source**: Select `main` branch
3. Under **Folder**: Select `/docs`
4. Click **Save**

### Step 3: Wait & Visit
- Wait 2-3 minutes for build
- Visit: `https://YOUR_USERNAME.github.io/YOUR_REPO/`

## ✅ Checklist

- [ ] All files in `docs/` folder
- [ ] Repository is public (or GitHub Pro for private)
- [ ] Pushed to `main` branch
- [ ] GitHub Pages enabled in settings
- [ ] Updated links in `content.md` Footer section

## 🔗 Update Your Links

Before deploying, edit `content.md` and update the Footer section:

```markdown
## Footer

**Tagline:** SHARINGAN-DEEP — Making small models see deeply into video

**Links:**
- [GitHub](https://github.com/YOUR_USERNAME/sharingan)
- [Paper](https://arxiv.org/YOUR_PAPER_ID)
- [Models](https://huggingface.co/YOUR_USERNAME/sharingan)
```

## 🎉 Done!

Your site is now live. To update:
1. Edit `docs/content.md`
2. Commit and push
3. GitHub Pages auto-rebuilds (2-3 min)

## 🌐 Custom Domain (Optional)

Want `sharingan.yourdomain.com`?

1. Add `CNAME` file in `docs/`:
   ```
   sharingan.yourdomain.com
   ```

2. Add DNS record at your domain provider:
   ```
   Type: CNAME
   Name: sharingan
   Value: YOUR_USERNAME.github.io
   ```

3. In GitHub Settings → Pages, add custom domain

---

**That's it!** Your beautiful documentation site is live.
