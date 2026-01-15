<!-- Language toggle (pure Markdown + inline HTML). Paste into BOTH files or a shared include. -->
<!-- Assumes you have two files: ./loop_cn.md and ./loop_en.md (rename paths as needed). -->

<div style="display:flex; gap:.5rem; align-items:center; margin: 0 0 1rem 0;">
  <strong>Language:</strong>
  <a id="lang-zh" href="./README.md">中文</a>
  <span>|</span>
  <a id="lang-en" href="./blog_en.md">English</a>
</div>

<script>
(function () {
  // Remember user choice
  var zh = document.getElementById('lang-zh');
  var en = document.getElementById('lang-en');
  if (!zh || !en) return;

  function setLang(lang){
    try { localStorage.setItem('blog_lang', lang); } catch(e){}
  }

  zh.addEventListener('click', function(){ setLang('zh'); });
  en.addEventListener('click', function(){ setLang('en'); });

  // If user previously chose a language, redirect to that file
  var saved = null;
  try { saved = localStorage.getItem('blog_lang'); } catch(e){}
  if (!saved) return;

  var path = window.location.pathname || "";
  var isZH = /README\.md$/.test(path);
  var isEN = /blog_en\.md$/.test(path);

  if (saved === 'zh' && !isZH) window.location.href = zh.getAttribute('href');
  if (saved === 'en' && !isEN) window.location.href = en.getAttribute('href');
})();
</script>
