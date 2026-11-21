// Colab ejecuta este código en el navegador que tiene acceso a la cámara.
(() => {
  window.cerrarManzanas?.();
  document.getElementById('visor-manzanas')?.remove();
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.style.cssText = 'position:absolute;width:1px;height:1px;opacity:0;pointer-events:none';
  const captura = document.createElement('canvas');
  const panel = document.createElement('div');
  const boton = document.createElement('button');
  const aviso = document.createElement('span');
  const medidas = document.createElement('img');
  const vista = document.createElement('canvas');
  let stream;
  let cerrada = false;
  let toma;
  let enVivo = false;
  let esperaDelModelo;

  function detener() {
    cerrada = true;
    clearTimeout(esperaDelModelo);
    stream?.getTracks().forEach(pista => pista.stop());
    video.pause();
    video.srcObject = null;
    boton.disabled = true;
    aviso.textContent = 'Cámara detenida';
  }

  async function abrir(dispositivo) {
    let nuevo;
    try {
      nuevo = await navigator.mediaDevices.getUserMedia({video: dispositivo, audio: false});
    } catch (error) {
      if (error.name === 'NotAllowedError') throw new Error('Permití el acceso a la cámara en el navegador y volvé a ejecutar la celda.');
      if (error.name === 'NotFoundError') throw new Error('El navegador no encontró una cámara conectada.');
      throw error;
    }
    if (cerrada) {
      nuevo.getTracks().forEach(pista => pista.stop());
      return;
    }
    stream = nuevo;
    stream.getVideoTracks()[0].addEventListener('ended', detener, {once: true});
  }

  window.cerrarManzanas = detener;
  window.addEventListener('pagehide', detener, {once: true});

  window.iniciarManzanas = async function(nombre, ancho, continuo = false) {
    enVivo = continuo;
    const pantalla = enVivo ? vista : medidas;
    pantalla.style.cssText = 'display:block;max-width:100%;height:auto;margin-top:6px';
    boton.textContent = 'Detener cámara';
    boton.onclick = detener;
    aviso.setAttribute('role', 'status');
    aviso.style.marginLeft = '10px';
    aviso.textContent = 'Abriendo cámara…';
    panel.id = 'visor-manzanas';
    panel.append(boton, aviso, video, pantalla);
    document.body.append(panel);
    await abrir({width: {ideal: ancho}});
    if (cerrada) return null;
    if (nombre.trim()) {
      const camaras = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
      const elegida = camaras.find(d => d.label.toLowerCase().includes(nombre.trim().toLowerCase()));
      if (!elegida) throw new Error('Cámaras disponibles: ' + camaras.map(d => d.label).join(', '));
      if (stream.getVideoTracks()[0].getSettings().deviceId !== elegida.deviceId) {
        stream.getTracks().forEach(pista => pista.stop());
        await abrir({deviceId: {exact: elegida.deviceId}, width: {ideal: ancho}});
      }
    }
    if (cerrada) return null;
    video.srcObject = stream;
    await video.play();
    if (cerrada) return null;
    captura.width = Math.min(ancho, video.videoWidth);
    captura.height = Math.max(1, Math.round(video.videoHeight * captura.width / video.videoWidth));
    pantalla.width = captura.width;
    pantalla.height = captura.height;
    if (enVivo) vista.getContext('2d').drawImage(video, 0, 0, vista.width, vista.height);
    window.google?.colab?.output?.setIframeHeight?.(document.documentElement.scrollHeight, true);
    aviso.textContent = 'Esperando la primera detección…';
    return [captura.width, captura.height];
  };

  window.leerManzanas = async function() {
    if (cerrada) return null;
    if (document.hidden) throw new Error('Mantené visible la pestaña de Colab.');
    await new Promise((resolver, rechazar) => {
      const limite = setTimeout(() => cerrada ? resolver() : rechazar(new Error('La cámara dejó de enviar cuadros.')), 3000);
      const listo = () => { clearTimeout(limite); resolver(); };
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(listo);
      else requestAnimationFrame(listo);
    });
    if (cerrada) return null;
    toma = performance.now();
    if (enVivo) {
      clearTimeout(esperaDelModelo);
      esperaDelModelo = setTimeout(detener, 90000);
    }
    captura.getContext('2d').drawImage(video, 0, 0, captura.width, captura.height);
    return captura.toDataURL('image/jpeg', 0.85);
  };

  // El notebook de evaluación devuelve la imagen ya dibujada y mide su latencia.
  window.mostrarManzanas = async function(jpeg) {
    if (cerrada) return null;
    medidas.src = jpeg;
    await medidas.decode();
    await new Promise(resolver => requestAnimationFrame(() => requestAnimationFrame(resolver)));
    if (cerrada) return null;
    aviso.textContent = 'Cámara en vivo';
    return performance.now() - toma;
  };

  // En vivo solo vuelven las cajas. Se dibujan sobre el mismo cuadro que recibió YOLO.
  window.siguienteManzanas = function(marcas) {
    if (cerrada) return null;
    const ctx = vista.getContext('2d');
    ctx.drawImage(captura, 0, 0);
    ctx.font = '15px sans-serif';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 2;
    for (const {caja, texto, color} of marcas) {
      const [x1, y1, x2, y2] = caja;
      ctx.strokeStyle = 'rgb(' + color.join(',') + ')';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const ancho = Math.min(vista.width, ctx.measureText(texto).width + 12);
      const x = Math.max(0, Math.min(x1, vista.width - ancho));
      const y = y1 >= 23 ? y1 - 23 : Math.min(y1 + 2, Math.max(0, vista.height - 22));
      ctx.fillStyle = '#191d25';
      ctx.fillRect(x, y, ancho, 22);
      ctx.fillStyle = 'white';
      ctx.fillText(texto, x + 6, y + 11, Math.max(1, ancho - 12));
    }
    aviso.textContent = marcas.length === 1 ? '1 manzana' : marcas.length + ' manzanas';
    return window.leerManzanas();
  };
})();
