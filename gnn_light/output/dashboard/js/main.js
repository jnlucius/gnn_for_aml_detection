document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

try{
  initLiveTab();
}catch(e){
  console.error('Live Stream tab failed to initialize', e);
}

try{
  initAnalystTab();
}catch(e){
  console.error('Analyst Review tab failed to initialize', e);
}
