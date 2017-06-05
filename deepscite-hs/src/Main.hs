{-# LANGUAGE OverloadedLists        #-}
{-# LANGUAGE ConstraintKinds        #-}
{-# LANGUAGE DataKinds              #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE OverloadedStrings      #-}
{-# LANGUAGE RankNTypes             #-}
{-# LANGUAGE ScopedTypeVariables    #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleContexts      #-}

module Main where


import qualified TensorFlow.GenOps.Core     as TFC 
import qualified TensorFlow.Minimize        as TF
import qualified TensorFlow.Ops             as TF hiding ( initializedVariable
                                                         , zeroInitializedVariable )
import qualified TensorFlow.Variable        as TF
import qualified TensorFlow.Types           as TF
import qualified TensorFlow.Tensor          as TF
import qualified TensorFlow.Build           as TF
import qualified TensorFlow.Output          as TF
import qualified TensorFlow.Core            as TF
import qualified TensorFlow.Gradient        as TF
import qualified TensorFlow.BuildOp         as TF
import qualified TensorFlow.EmbeddingOps    as TF
import qualified TensorFlow.NN              as TF

import           Data.Csv                   ( FromNamedRecord(..)
                                            , decodeByName
                                            , (.:)
                                            )
import qualified Data.Vector                as V
import           Data.Int                   (Int32, Int64)
import           Data.Word                  (Word16)
import           Data.Monoid                (mconcat)
import           Data.List                  (genericLength)
import           Control.Monad              (join)
import           Control.Monad.IO.Class     (liftIO)
import           Lens.Family2               ((.~), (&))
import           Control.Monad              ( zipWithM
                                            , when
                                            , forM
                                            , forM_
                                            )
import qualified Data.ByteString.Lazy       as BL
import           Data.ByteString            (ByteString)
import           Control.Applicative        ( (<*>)
                                            , (<$>)
                                            )
import           Data.Text                  ( Text(..)
                                            , splitOn
                                            , unpack
                                            )
import           System.FilePath            ( (</>)
                                            , (<.>))

import Data.RVar (runRVar)
import Data.Random.Extras (shuffle)
import Data.Random.RVar ( RVar(..)
                        )
import Data.Random.Extras
import Data.Random.Source.Std
import Data.Random.Source.StdGen
import Control.Monad.State
import Data.Random hiding (shuffle)
import Data.ProtoLens (def)
import Proto.Tensorflow.Core.Framework.Tensor
    ( TensorProto
    , dtype
    , tensorShape
    )
import qualified Proto.Tensorflow.Core.Framework.TensorShape
  as TensorShape
import Text.Printf (printf)



trainDataDir :: FilePath
trainDataDir = "../data/noon/"


inputSize        = 200   :: Int64 
batchSize        = 100   :: Int
convFeatures     = 1     :: Int64
minibatchSize    = -1    :: Int64
embeddedWordSize = 150   :: Int64
vocabSize        = 79951 :: Int64
convSize         = 3     :: Int64
stride           = 1     :: Int64



conv2D :: forall v1 v2 t m . ( TF.MonadBuild m
                             , TF.TensorType t
                             , TF.TensorKind v1
                             , TF.TensorKind v2
                             , TF.Rendered (TF.Tensor v1)
                             , TF.OneOf '[Data.Word.Word16, Double, Float] t)
       => TF.Tensor v1 t            -- ^ __input__
       -> TF.Tensor v2 t            -- ^ __filter__
       -> [Int64]                   -- ^ __strides__
       -> m (TF.Tensor TF.Value t)  -- ^ __output__
conv2D input filter strides = TF.render $ do
            TFC.conv2D' ( 
                      (TF.opAttr "strides"      .~ (strides :: [Int64]))
                    . (TF.opAttr "padding"      .~ ("SAME"  :: ByteString))
                    . (TF.opAttr "data_format"  .~ ("NHWC"  :: ByteString))
                ) input filter


randomParam :: TF.Shape -> TF.Session (TF.Tensor TF.Value Float)
randomParam (TF.Shape shape) = TF.truncatedNormal (TF.vector shape)


reduceMean :: TF.Tensor v Float -> TF.Tensor TF.Build Float
reduceMean xs = TF.mean xs (TF.range 0 (TFC.rank xs) 1)


data Article = Article {
      uid              :: !Text
    , titleIds         :: !Text
    , abstractIds      :: !Text
    , sciteProbability :: !Float
    } deriving (Show)


readArticles :: FilePath -> IO (V.Vector Article)
readArticles dataset = do
    csvData <- BL.readFile (trainDataDir </> dataset <.> "csv") 
    case decodeByName csvData of
        Left err     -> error $ "Error reading csv: " ++ (show err)
        Right (_, v) -> return v


instance FromNamedRecord Article where
    parseNamedRecord r = Article <$> 
            r .: "id"
        <*> r .: "wordset_1_ids"
        <*> r .: "wordset_2_ids"
        <*> r .: "probability"


say = liftIO . putStrLn

main = TF.runSession $ do
    say "Loading and shuffling data ..."

    trainArticles'      <- fmap V.toList $ liftIO (readArticles "train")
    validationArticles' <- fmap V.toList $ liftIO (readArticles "validation")

    trainArticles       <- liftIO (runRVar (shuffle trainArticles'     ) StdRandom :: IO [Article])
    validationArticles  <- liftIO (runRVar (shuffle validationArticles') StdRandom :: IO [Article])
     

    ---------------------------------------------------------------------
    --
    say "Building model ..."

    titles      :: TF.Tensor TF.Value Int32 <- TF.placeholder [minibatchSize, inputSize, 1]
    probs       :: TF.Tensor TF.Value Float <- TF.placeholder [minibatchSize]

    embedding   :: TF.Variable Float   <- TF.initializedVariable =<< randomParam [vocabSize, embeddedWordSize]

    -- Why?
    em' <- TF.render $ TF.readValue embedding
    wordVectors :: TF.Tensor TF.Value Float <- TF.embeddingLookup [em'] titles

    convWeights :: TF.Variable Float   <- TF.initializedVariable =<< randomParam [convSize, 1, embeddedWordSize, convFeatures]
    convBias    :: TF.Variable Float   <- TF.zeroInitializedVariable [convFeatures]

    conv <- conv2D wordVectors (TF.readValue convWeights) [1, stride, 1, 1] 
    let convOut = conv `TFC.add` (TF.readValue convBias)
        means   = TF.mean convOut (TF.vector [1,2,3 :: Int32])

    means' <- TF.render means

    sigmoidLoss <- TF.sigmoidCrossEntropyWithLogits means' probs
    lossStep <- TF.render $ reduceMean sigmoidLoss


    -- Inference
    let finalProbs = TFC.sigmoid means
    predict  <- TF.render finalProbs

    -- Accuracy
    let rightEnough  = TF.equal probs (TFC.cast (TFC.greater finalProbs (TF.scalar 0.5)))
        accuracyStep = reduceMean (TFC.cast rightEnough)


    let params = [convWeights, convBias, embedding] :: [TF.Variable Float]

    trainStep <- TF.minimizeWith TF.adam lossStep params


    let train    = \ts ps -> TF.runWithFeeds_ [TF.feed titles ts, TF.feed probs ps] trainStep
        loss     = \ts ps -> TF.unScalar <$> TF.runWithFeeds [TF.feed titles ts, TF.feed probs ps] lossStep
        accuracy = \ts ps -> TF.unScalar <$> TF.runWithFeeds [TF.feed titles ts, TF.feed probs ps] accuracyStep
    --
    ---------------------------------------------------------------------

    let -- TODO: Better yet, have some appropriate method of data
        -- selection here, including epoch's and what-not.
        selectBatch i xs = take batchSize $ drop (i * batchSize) (cycle xs)

    say "Training ..."
    forM_ ([0..1000] :: [Int]) $ \i -> do
        let batch              = selectBatch i trainArticles

            idStrToIds :: Article -> [Int32]
            idStrToIds article = fromIntegral . read . unpack <$> splitOn " " (titleIds article)

            imgDim  = [fromIntegral batchSize, inputSize, 1] :: TF.Shape
            probDim = [fromIntegral batchSize]               :: TF.Shape
            flatten = join

            -- We need to append zeros so that the numbers here aren't random
            -- indicies we look up in the embedding.
            
            wordIds = idStrToIds <$> batch
            wordIdsWithZeros = fmap (\xs -> xs ++ replicate (fromIntegral inputSize - length xs) 0) wordIds

            wordIdsFlat = V.fromList $ flatten wordIdsWithZeros
            sciteProbs  = V.fromList $ sciteProbability <$> batch

            titles = TF.encodeTensorData imgDim  wordIdsFlat :: TF.TensorData Int32
            probs  = TF.encodeTensorData probDim sciteProbs  :: TF.TensorData Float

        -- Train
        train titles probs

        when (i `mod` 100 == 0) $ do

            -- paramValues <- mapM TF.render (map TF.readValue params)
            -- TF.save "./checkpoint" paramValues >>= TF.run_

            modelLoss <- loss titles probs
            accuracy  <- accuracy titles probs

            liftIO $ do
                putStrLn ""
                putStrLn $ "         i: " ++ show i
                putStrLn $ "      loss: " ++ show modelLoss
                putStrLn $ "  accuracy: " ++ show accuracy

    liftIO $ putStrLn "Done"
