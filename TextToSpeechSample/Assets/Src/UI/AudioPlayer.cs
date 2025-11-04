using System;
using System.Threading.Tasks;
using Unity.AppUI.Redux;
using Unity.AppUI.UI;
using Unity.InferenceEngine.Samples.TTS.State;
using Unity.InferenceEngine.Samples.TTS.Utils;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.TTS.UI
{
    [UxmlElement]
    public partial class AudioPlayer : VisualElement
    {
        AppStoreManager m_StoreManager;

        AudioSource AudioSource
        {
            get
            {
                if (m_AudioSource == null)
                {
                    var go = GameObject.Find("App");
                    if (go == null)
                    {
                        go = new GameObject("App")
                        {
                            hideFlags = HideFlags.HideAndDontSave
                        };

                        go.AddComponent<AudioSource>();
                    }

                    m_AudioSource = go.GetComponent<AudioSource>();
                }
                return m_AudioSource;
            }
        }


        AudioSource m_AudioSource;

        IconButton m_PlayButton;
        IconButton m_SaveButton;
        Text m_TimeStamp;
        SliderFloat m_AudioProgress;

        public AudioPlayer()
        {
            RegisterCallback<AttachToPanelEvent>(OnAttachToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);
        }

        void OnAttachToPanel(AttachToPanelEvent evt)
        {
            m_PlayButton = this.Q<IconButton>(classes: "play-button");
            m_PlayButton.clickable.clicked += OnPlayButtonClicked;
            m_SaveButton = this.Q<IconButton>(classes: "download-button");
            m_SaveButton.clickable.clicked += OnSaveButtonClicked;
            m_TimeStamp = this.Q<Text>(classes: "time-text");
            m_AudioProgress = this.Q<SliderFloat>(classes: "progress-bar");
            m_AudioProgress.RegisterValueChangingCallback(OnSliderValueChanging);

            m_AudioProgress.RegisterCallback<PointerDownEvent>(_ => AudioSource.Pause(), TrickleDown.TrickleDown);
            m_AudioProgress.RegisterCallback<PointerUpEvent>(_ =>  Play(), TrickleDown.TrickleDown);
        }

        void OnSliderValueChanging(ChangingEvent<float> evt)
        {
            AudioSource.time = evt.newValue / 100f * AudioSource.clip.length;
            UpdateVisuals();
        }

        void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            m_PlayButton.clickable.clicked -= OnPlayButtonClicked;
            m_SaveButton.clickable.clicked -= OnSaveButtonClicked;
        }

        void OnPlayButtonClicked()
        {
            if(AudioSource.isPlaying)
                AudioSource.Pause();
            else
                Play();
        }

        async Task Play()
        {
            AudioSource.Play();
            while (AudioSource.isPlaying)
            {
                UpdateVisuals();
                await Task.Yield();
            }

            UpdateVisuals();
        }

        public void Initialize(AppStoreManager storeManager)
        {
            m_StoreManager = storeManager;
            m_StoreManager.Store.Subscribe( AppSlice.Name, (AppState state) => state.AudioClip, OnAudioClipChanged);
        }

        void OnAudioClipChanged(AudioClip clip)
        {
            AudioSource.clip = clip;
            UpdateVisuals();
        }

        void UpdateVisuals()
        {
            m_AudioProgress.SetValueWithoutNotify(AudioSource.time / AudioSource.clip.length * 100f);
            m_TimeStamp.text = $"{AudioSource.time:0:00} / {AudioSource.clip.length:0:00}";
            m_PlayButton.icon = AudioSource.isPlaying ? "pause" : "play";

            m_SaveButton.SetEnabled(Application.isEditor);
        }

        void OnSaveButtonClicked()
        {
            #if UNITY_EDITOR
            //Open save file dialog
            var path = EditorUtility.SaveFilePanel("Save Audio", "", "audio.wav", "wav");
            if (string.IsNullOrEmpty(path))
                return;

            var state = m_StoreManager.Store.GetState<AppState>(AppSlice.Name);
            WavUtils.WriteFloatWav(path, state.Waveform.DownloadToArray());
            #endif
        }
    }
}
