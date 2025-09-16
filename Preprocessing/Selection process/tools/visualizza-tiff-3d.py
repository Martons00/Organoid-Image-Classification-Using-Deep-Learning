def visualizza_tiff_3d(image_path, filename):
    """
    Visualizza un file TIFF con i 3 piani ortogonali (XY, YZ, XZ) per volumi 3D
    Per file 2D mostra l'immagine singola con informazioni aggiuntive
    
    Args:
        image_path (str): Percorso del file TIFF
        filename (str): Nome del file per il titolo
    """
    try:
        with Image.open(image_path) as img:
            # Carica tutto il volume se è multi-frame
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                print(f"    📚 Caricamento volume 3D ({img.n_frames} frame)...")
                
                # Carica tutti i frame in un array 3D
                frames = []
                for i in range(img.n_frames):
                    img.seek(i)
                    frame = np.array(img)
                    frames.append(frame)
                
                # Crea array 3D: (depth, height, width)
                volume_3d = np.stack(frames, axis=0)
                depth, height, width = volume_3d.shape
                
                print(f"    📦 Dimensioni volume: {depth}x{height}x{width} (D×H×W)")
                
                # Calcola slice centrali per ogni piano
                z_center = depth // 2
                y_center = height // 2  
                x_center = width // 2
                
                # Estrai i tre piani ortogonali
                slice_xy = volume_3d[z_center, :, :]      # Piano XY (slice lungo Z)
                slice_yz = volume_3d[:, y_center, :]      # Piano YZ (slice lungo Y)  
                slice_xz = volume_3d[:, :, x_center]      # Piano XZ (slice lungo X)
                
                # Crea figura con 3 subplot
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Piano XY (assiale)
                vmin_xy, vmax_xy = slice_xy.min(), slice_xy.max()
                im1 = axes[0].imshow(slice_xy, cmap='gray', vmin=vmin_xy, vmax=vmax_xy)
                axes[0].set_title(f'Piano XY (Assiale)\\nSlice Z={z_center}/{depth-1}\\nmin={vmin_xy}, max={vmax_xy}')
                axes[0].set_xlabel('X (Width)')
                axes[0].set_ylabel('Y (Height)')
                if vmax_xy > vmin_xy:
                    plt.colorbar(im1, ax=axes[0], shrink=0.8)
                
                # Piano YZ (sagittale)  
                vmin_yz, vmax_yz = slice_yz.min(), slice_yz.max()
                im2 = axes[1].imshow(slice_yz, cmap='gray', vmin=vmin_yz, vmax=vmax_yz, aspect='auto')
                axes[1].set_title(f'Piano YZ (Sagittale)\\nSlice X={x_center}/{width-1}\\nmin={vmin_yz}, max={vmax_yz}')
                axes[1].set_xlabel('Z (Depth)')
                axes[1].set_ylabel('Y (Height)')
                if vmax_yz > vmin_yz:
                    plt.colorbar(im2, ax=axes[1], shrink=0.8)
                
                # Piano XZ (coronale)
                vmin_xz, vmax_xz = slice_xz.min(), slice_xz.max()
                im3 = axes[2].imshow(slice_xz, cmap='gray', vmin=vmin_xz, vmax=vmax_xz, aspect='auto')
                axes[2].set_title(f'Piano XZ (Coronale)\\nSlice Y={y_center}/{height-1}\\nmin={vmin_xz}, max={vmax_xz}')
                axes[2].set_xlabel('X (Width)')
                axes[2].set_ylabel('Z (Depth)')
                if vmax_xz > vmin_xz:
                    plt.colorbar(im3, ax=axes[2], shrink=0.8)
                
                # Statistiche del volume completo
                vol_min, vol_max = volume_3d.min(), volume_3d.max()
                vol_mean = volume_3d.mean()
                vol_std = volume_3d.std()
                
                plt.suptitle(f'Volume 3D: {filename}\\n'
                           f'Shape: {depth}×{height}×{width} | '
                           f'Range: [{vol_min}, {vol_max}] | '
                           f'Mean: {vol_mean:.2f} ± {vol_std:.2f}', 
                           fontsize=14, fontweight='bold')
                
                print(f"    📊 Statistiche volume completo:")
                print(f"        Range: [{vol_min}, {vol_max}]")
                print(f"        Media: {vol_mean:.3f} ± {vol_std:.3f}")
                print(f"        Slice centrali: XY={z_center}, YZ={x_center}, XZ={y_center}")
                
            else:
                # File 2D singolo - mostra con informazioni aggiuntive
                img_array = np.array(img)
                
                if len(img_array.shape) == 2:
                    # Immagine 2D grayscale
                    height, width = img_array.shape
                    
                    # Crea figura con 3 subplot: immagine + istogramma + profili
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Immagine principale
                    vmin, vmax = img_array.min(), img_array.max()
                    im = axes[0].imshow(img_array, cmap='gray', vmin=vmin, vmax=vmax)
                    axes[0].set_title(f'Immagine 2D\\n{height}×{width}\\nmin={vmin}, max={vmax}')
                    axes[0].set_xlabel('X (Width)')
                    axes[0].set_ylabel('Y (Height)')
                    if vmax > vmin:
                        plt.colorbar(im, ax=axes[0], shrink=0.8)
                    
                    # Istogramma intensità
                    axes[1].hist(img_array.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
                    axes[1].set_title('Istogramma Intensità')
                    axes[1].set_xlabel('Valore Pixel')
                    axes[1].set_ylabel('Frequenza')
                    axes[1].grid(True, alpha=0.3)
                    
                    # Profili centrali (riga e colonna centrale)
                    center_row = height // 2
                    center_col = width // 2
                    
                    axes[2].plot(img_array[center_row, :], 'b-', label=f'Riga centrale ({center_row})', alpha=0.8)
                    axes[2].plot(img_array[:, center_col], 'r-', label=f'Colonna centrale ({center_col})', alpha=0.8)
                    axes[2].set_title('Profili Centrali')
                    axes[2].set_xlabel('Posizione Pixel')
                    axes[2].set_ylabel('Intensità')
                    axes[2].legend()
                    axes[2].grid(True, alpha=0.3)
                    
                    plt.suptitle(f'Immagine 2D: {filename}', fontsize=14, fontweight='bold')
                    
                elif len(img_array.shape) == 3:
                    # Immagine RGB o multi-canale
                    height, width, channels = img_array.shape
                    
                    fig, axes = plt.subplots(1, min(3, channels), figsize=(6 * min(3, channels), 6))
                    if channels == 1:
                        axes = [axes]
                    
                    for i in range(min(3, channels)):
                        channel_data = img_array[:, :, i]
                        vmin, vmax = channel_data.min(), channel_data.max()
                        
                        im = axes[i].imshow(channel_data, cmap='gray', vmin=vmin, vmax=vmax)
                        axes[i].set_title(f'Canale {i}\\nmin={vmin}, max={vmax}')
                        axes[i].axis('off')
                        
                        if vmax > vmin:
                            plt.colorbar(im, ax=axes[i], shrink=0.8)
                    
                    plt.suptitle(f'Immagine Multi-canale: {filename}\\n'
                               f'Shape: {height}×{width}×{channels}', 
                               fontsize=14, fontweight='bold')
                
                print(f"    📊 Immagine 2D: shape={img_array.shape}")
                print(f"        Range: [{img_array.min()}, {img_array.max()}]")
                print(f"        Media: {img_array.mean():.3f} ± {img_array.std():.3f}")
            
            plt.tight_layout()
            plt.show()
                
    except Exception as e:
        print(f"    ❌ Errore nella visualizzazione: {str(e)}")
        print("    💡 Il file potrebbe essere corrotto o in formato non supportato")
        
        # Fallback: prova a mostrare almeno le info base
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'n_frames'):
                    print(f"    📋 Info file: {img.n_frames} frame, mode={img.mode}")
                else:
                    print(f"    📋 Info file: {img.size}, mode={img.mode}")
        except:
            print("    ❌ Impossibile leggere anche le informazioni base del file")